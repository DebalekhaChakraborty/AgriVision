# Phase 4 — AWS deployment and observability

**Status: COMPLETE — UNCOMMITTED.** The service is deployed and running.

**Live endpoint:** https://yp2ajauzkm.us-east-1.awsapprunner.com

```
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/health
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/ready
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/version
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/metrics
POST https://yp2ajauzkm.us-east-1.awsapprunner.com/inspect              (multipart image)
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/inspection/{run_id}
GET  https://yp2ajauzkm.us-east-1.awsapprunner.com/inspection/{run_id}/trace
```

No vision code changed in this phase. The orchestrator, the policies and every
threshold are exactly what Phase 3 and Phase 3b committed; this phase puts an
HTTP surface, a container and an AWS runtime around them.

---

## 1. Architecture

Diagram: [`DEPLOYED_ARCHITECTURE.md`](DEPLOYED_ARCHITECTURE.md).

App Runner, ECR, S3, DynamoDB, CloudWatch, IAM. That is the whole list.

**Not used, and why.** API Gateway adds a second HTTP layer in front of a
service that already terminates TLS. Lambda cannot hold a 12 MB ONNX graph warm
without provisioned concurrency, which costs more than one small always-on
instance. Step Functions, SQS and EventBridge orchestrate across services, and
this loop is bounded, synchronous and lives inside one process. SageMaker hosts
models that `cv2.dnn` already runs. A VPC with NAT would add roughly the cost of
the whole service for no isolation this needs. Cognito guards an API that has no
user accounts.

**App Runner was chosen** because it supplies managed HTTPS with a certificate,
a health-check loop, rolling deployment and autoscaling, for one resource
definition and no load balancer. It was not chosen for cold-start behaviour, and
"serverless" is not claimed as a performance property — one instance is kept
provisioned precisely so there is no cold start.

**CloudFormation, not CDK.** The stack is four resources and two roles. A
declarative template needs no build step, no bootstrap stack and no Node
toolchain in the deployment path. CDK would earn its place if the topology grew
conditional; it has not.
[`infrastructure/aws/agrivision-stack.yaml`](../../infrastructure/aws/agrivision-stack.yaml)

---

## 2. The deployment failure worth reading

The first three deployments failed identically: image pulled successfully, then
`CREATE_FAILED`, with **no application log group created at all**. No
application logs means the container never ran, which rules out every
application-level explanation.

Diagnosis by bisection against a minimal `python:3.11-slim` image that deployed
successfully:

| Variant | Result |
| --- | --- |
| Minimal python image | **RUNNING** |
| AgriVision, no Docker `HEALTHCHECK` | CREATE_FAILED |
| AgriVision, no instance role, no env vars | CREATE_FAILED |
| AgriVision as **root** | **RUNNING** |
| AgriVision non-root with `--shell /bin/sh` | **RUNNING** |

**Root cause: `useradd --shell /usr/sbin/nologin`.** App Runner needs to exec a
shell for the container user. The image ran perfectly under `docker run`
locally, which is exactly why this was expensive to find: nothing on the host
reproduces it.

Non-root is kept — the fix was the login shell, not the security property. Ruled
out along the way: the Docker `HEALTHCHECK`, the instance role, the runtime
environment variables, the manifest media type and the architecture.

**A second failure followed.** The service went live with the model downloaded
and its SHA-256 verified, and still reported `model_loaded: false`.
`load_condition_model` also requires `manifest.json` — it checks the
preprocessing contract against the source checkpoint and refuses to load without
it, which is correct, since a graph without its preprocessing description can be
fed inputs it was never trained on. Both objects are now fetched.

---

## 3. Container

| Property | Value |
| --- | --- |
| Base | `python:3.11-slim-bookworm` pinned by digest `sha256:a36c24f9…` |
| Size | **719 MB** |
| Build | 56 s, multi-stage |
| User | non-root, uid 10001 |
| Platform | linux/amd64 |

**Absent from the serving image**, each verified by attempting the import inside
it: torch, torchvision, onnx, onnxruntime, matplotlib, pytest, tensorflow,
keras, transformers, jupyter. Inference runs through `cv2.dnn`, so the serving
path never imports a training framework.

The image grew from 506 MB to 719 MB when `libxcb1` and `libgl1` were added.
`opencv-python` links against X and GL even when no window is ever created, and
the first build failed at `import cv2` inside the container while the host was
fine. `opencv-python-headless` would avoid them, and is deliberately **not**
used: every phase from 1 onward verified `opencv-python 5.0.0.93` specifically,
and swapping the wheel would mean the deployed binary is not the one the
calibration was measured against.

---

## 4. OpenCV 5 in the container

Closing a Phase 0 unknown. Verified **inside** the running image:
`cv2.__version__ == 5.0.0`, plus `cvtColor` (three conversions), `Laplacian`,
`Sobel`, `GaussianBlur`, `CLAHE`, Otsu thresholding, `morphologyEx`,
`connectedComponentsWithStats`, `findContours`, `convexHull`,
`convexityDefects`, `distanceTransform`, `grabCut`, `imencode`/`imdecode`,
`dnn.blobFromImage`, `normalize` and `dft` — followed by the competition
perception path itself and a locked-policy load returning fingerprint
`15b7fb0908d6d622`.

---

## 5. Model artifact lifecycle

```
S3 (private, encrypted, versioned)
   → startup download via the instance role
   → SHA-256 compared against configuration
   → cv2.dnn, only on a match
```

The expected hash
(`77d8614671873cb41ebfd725f0a4b468beca9c30cfe6a3361217343917324d4b`) is
compiled into the image and overridable by an operator. It is never read from
the same place as the artifact: a checksum shipped alongside the file it checks
proves only that the file arrived intact.

**A mismatch is fatal.** Not a warning and not a degraded mode — the service
stays unready and says why. A classifier running on an unverified graph is worse
than one that refuses to start, because its answers look exactly like correct
ones. There is no code path that downloads a model from a URL supplied at
request time.

The ONNX binary is not committed: it derives from a research checkpoint whose
source dataset licence is recorded as Unknown with no redistribution grant.

---

## 6. API and input handling

A filename extension is a claim by the client, not a fact about the bytes.
Validation runs on content: magic-byte identification, then a real decode, then
dimension and pixel-count bounds. Limits exist to bound work — an unbounded
upload is memory exhaustion, and a 40000×40000 PNG is a denial of service
dressed as a photograph.

The input contract is returned in **every** inspection response:

> Submit one primary fruit per inspection image. The system does not extract a
> single item from market stalls, piles, crates or trees carrying multiple
> fruits. This is a scope boundary, not a food-safety statement.

Status semantics are in [`DEPLOYED_ARCHITECTURE.md`](DEPLOYED_ARCHITECTURE.md).

---

## 7. Evidence

### Local container

| Scenario | HTTP | Final state | Model ran |
| --- | ---: | --- | --- |
| normal | 200 | COMPLETE | yes |
| underexposed | 200 | REQUEST_RECAPTURE | no |
| severe blur | 200 | REQUEST_RECAPTURE | no |
| foreground failure | 200 | REQUEST_RECAPTURE | no |

### AWS

Identical outcomes on the deployed service. Trace retrieved from DynamoDB with
the causal chain intact — `high_frequency_ratio 0.272266` against floor
`0.318198` → `REQUEST_RECAPTURE`. Leak scan on the retrieved trace: none.

| Rejected input | HTTP | Reason code |
| --- | ---: | --- |
| plain text | 415 | `UNSUPPORTED_MEDIA_TYPE` |
| truncated JPEG | 400 | `UNDECODABLE_IMAGE` |
| 13 MB upload | 413 | `UPLOAD_TOO_LARGE` |

`unsafe_inference_count` — the classifier running on a refused capture — is
**0** locally and on AWS.

### A transport finding — and what was done about it

Scenarios are uploaded as JPEG, which is what a client actually sends, and JPEG
is lossy. `darken(reference, 0.12)` completes in-process; after a JPEG round
trip it has gamma applied and accepted and is then refused by the severe
contrast check. Pre-remediation metrics are near-identical (luminance 0.0480 vs
0.0485); the divergence is post-gamma contrast falling below the 0.12 severe
limit.

The agent is responding correctly to the pixels it was given. **No threshold was
loosened to hide this** — that would trade a calibrated limit for a tidier demo.

Instead the service now reports `lossless_transport` on every response, and PNG
is supported for the cases where pixel identity matters. A test asserts that a
PNG upload of `darken(reference, 0.12)` reaches exactly the same terminal state
and the same content hash as inspecting the array in-process, so any future
divergence is attributable to the service rather than to the transport.

JPEG results are **transport-domain behaviour**, not a defect, and are reported
as such.

---

## 8. Observability

One JSON object per line to CloudWatch Logs, with `run_id`, `event`, `state`,
`tool`, `action`, `reason_code`, `evidence_maturity`, `duration_ms`, `success`
and `pipeline_version`. Verified against live CloudWatch events.

Redaction is enforced on every value recursively rather than per call site,
because the one call site that forgets is the one that matters. Never logged:
image bytes, credentials, authorization headers, filesystem paths, upload
filenames, full request bodies. A value containing a path or credential marker
is replaced **entirely** — a partial mask still leaks length and prefix.

`/metrics` exposes twelve counters and five duration series.
**No CloudWatch custom metrics**: they bill per metric per month, App Runner
already publishes request count, latency and error rate for free, and the agent
dimension those miss lives in the structured logs where Logs Insights can query
it without a second billing surface. No production SLO is defined — there is no
traffic history to base one on.

---

## 9. Trace persistence

DynamoDB, on-demand, 14-day TTL. **Not** local disk: App Runner replaces
instances freely and may run more than one, so a run inspected on one instance
must be retrievable from another.

A record carries content hashes, metrics, flags, actions, timings and policy
fingerprints — and the **full step list, not a prose summary**, so a retrieved
trace still shows the OpenCV metric, the threshold, the decision and the tool
that decision invoked. Verified record size 8,464 bytes. Never stored: image
bytes, EXIF, paths, credentials, upload filename.

**A persistence failure never changes a result.** The perception and decision
work has already happened and is already correct; losing the ability to store it
is an availability problem for later retrieval, not a reason to invent, suppress
or alter a verdict. Writes fail soft, are counted, and report
`X-Trace-Stored: false` while the body stays identical.

---

## 10. Security and IAM

The runtime role can read one prefix of one bucket and write to one table. It
cannot list buckets, cannot delete anything, and cannot reach any other service.
No `AdministratorAccess`. No long-lived credential in the image, its environment
or the repository — the container receives a task role.

| Control | State |
| --- | --- |
| HTTPS | managed certificate via App Runner |
| S3 public access | fully blocked; AES256; versioned |
| DynamoDB | encryption at rest enabled |
| Upload limits | 12 MB, 12000 px, 50 M pixels |
| Content validation | magic bytes + real decode, never the filename |
| Error bodies | reason codes only; no internals, no tracebacks |
| Dependency pinning | exact versions; base image pinned by digest |
| Artifact verification | SHA-256, fail closed |
| Secret scan | clean |

The scan flagged one item: `GPG_KEY` in the image environment. It is the
upstream Python base image's **public** key fingerprint for verifying the Python
tarball, not a secret.

### The deployment identity

The first Phase 4 deployment was performed while the CLI resolved to the account
root. That is not an acceptable documented provisioning path, and it has been
replaced.

**The reproducible path is now `user/agrivision-deployer`**, holding one
customer-managed policy, `AgriVisionDeploymentAccess`. It can:

- push to and read the `agrivision` ECR repository, and nothing else in ECR
- create and update the `agrivision-inspection` App Runner service
- manage the model-artifact bucket and objects under `models/*`
- manage the `agrivision-inspection-traces` table
- read `/aws/apprunner/*` log groups
- `PassRole` the two App Runner service roles, by ARN
- create the App Runner service-linked role, conditioned on that service name

Three statements use `Resource: "*"` and each is forced:
`ecr:GetAuthorizationToken` has no resource form,
`apprunner:CreateService`/`ListServices` act before a service ARN exists, and
`iam:CreateServiceLinkedRole` is constrained instead by an
`iam:AWSServiceName` condition. There is no `AdministratorAccess` and no
wildcard action anywhere.

Least privilege was verified in both directions: as the deployer, `describe-service`
and `ecr describe-images` succeed, while `iam list-users` and `s3 ls` both return
`AccessDenied`.

**A role was preferred and proved impossible from here.** `AgriVisionDeployer`
exists, carries the same managed policy, and is trusted by the account — but
`sts:AssumeRole` from root fails outright with *"Roles may not be assumed by
root accounts."* That is an AWS constraint, not a configuration mistake, and it
is exactly the condition under which a dedicated user is the correct fallback.
A future operator who is already an IAM principal should assume the role; the
user `agrivision-deployer` also holds `sts:AssumeRole` on it for that path.

**Root credentials.** `AccountAccessKeysPresent` is **0** — no persistent root
access keys exist on this account, and root MFA is enabled. The session used for
the initial deployment was temporary root credentials, so there is nothing to
delete or rotate. Nothing was disabled automatically.

**Verified after remediation:** the image was pushed and the deployment
triggered entirely as `agrivision-deployer`, the service returned to `RUNNING`,
and `/ready` reports OpenCV 5.0.0, the model artifact verified and loaded, and
DynamoDB healthy. Root is no longer required for any normal deployment
operation.

---

## 11. Latency

| Path | Median | p95 |
| --- | ---: | ---: |
| Complete with inference | 297 ms | 313 ms |
| Recapture terminal | 241 ms | 268 ms |
| Segmentation failure | 244 ms | 259 ms |

**Client-observed, over the public internet, including TLS and the multipart
upload.** Not a server-side compute measurement and not comparable to the
in-process figures in `results/phase3/latency.json`. Local container: 229 ms for
a complete inspection.

---

## 12. Cost

Order of magnitude: **single-digit to low-tens of US dollars per month** for a
continuously running demonstration. The dominant term is provisioned App Runner
memory while idle; one 11 MB S3 object, a few thousand small DynamoDB items, one
ECR image and modest JSON logs are negligible beside it.

No precise figure is given because it would be invented — actual cost depends on
current published rates, uptime and traffic, none observed over a full billing
period. Controls: on-demand DynamoDB, 14-day TTL, no upload retention, one
instance. Deleting the App Runner service stops essentially all of it.

---

## 13. Reproduction

Provision as the scoped deployer, never as root:

```bash
export AWS_ACCESS_KEY_ID=...        # agrivision-deployer
export AWS_SECRET_ACCESS_KEY=...
aws sts get-caller-identity          # must NOT print :root
```

```bash
# 1. resolve dependencies
.venv-competition/bin/pip install -r requirements-competition.txt

# 2. tests
.venv-competition/bin/python -m pytest tests/competition/ -q

# 3. build and verify the container
docker build -t agrivision:phase4 .
docker run --rm agrivision:phase4 python -c "import cv2; print(cv2.__version__)"

# 4. run locally (model mounted)
docker run -d -p 8080:8080 \
  -v "$(pwd)/competition/models/artifacts/mobilenetv3_large_v2exp004:/opt/agrivision/model:ro" \
  agrivision:phase4
curl localhost:8080/ready

# 5. provision AWS
aws cloudformation deploy --template-file infrastructure/aws/agrivision-stack.yaml \
  --stack-name agrivision --capabilities CAPABILITY_NAMED_IAM

# 6. publish the model artifact and the image
aws s3 cp <model.onnx>   s3://agrivision-model-artifacts-<account>/models/mobilenetv3_large_v2exp004/
aws s3 cp <manifest.json> s3://agrivision-model-artifacts-<account>/models/mobilenetv3_large_v2exp004/
docker push <account>.dkr.ecr.<region>.amazonaws.com/agrivision:phase4

# 7. smoke-test
curl https://<service-url>/ready
```

The model artifact must be obtained separately; it is not in the repository.

---

## 14. Limitations

**The deployer's access key lives on the operator's machine.** A dedicated user
with a long-lived key is weaker than a short-lived assumed-role session; it was
chosen only because root cannot assume a role. On a machine with an existing IAM
identity, assume `AgriVisionDeployer` instead and no long-lived key is needed.

**One instance, one worker.** No load testing has been done and no concurrency
behaviour is characterised.

**No authentication.** The endpoint is public. Acceptable for a judged demo,
not for anything else.

**In-memory metrics.** `/metrics` reports one instance's counters and resets on
replacement. Durable aggregate metrics would need CloudWatch custom metrics,
deliberately not adopted (§8).

**No autoscaling evidence.** Scaling is App Runner's default; nothing here
exercised or measured it.

**Cost is an order of magnitude, not a figure** (§12).

**The JPEG transport finding** (§7) means API-path outcomes can differ from
in-process results on marginal captures.

**No judge UI.** The API returns JSON; a browsable demo is Phase 5.

---

## 15. Claim boundary

Supported:

> A bounded OpenCV 5 Agentic Vision inspection loop runs as a container on AWS
> App Runner, fetching and verifying its model from S3, persisting causal traces
> to DynamoDB and emitting structured per-step logs to CloudWatch.

**Not** claimed: any production readiness, any SLO, any load characteristic, any
cost figure to the dollar, and any food-safety, edibility or contamination
judgement. The service reports **visible condition** only.
