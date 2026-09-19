# AgriVision — deployed architecture

Every component below is deployed and exercised by the live service. Nothing is
aspirational, and nothing is included because it would look good on a diagram.

**Not present, on purpose:** Bedrock, SageMaker, API Gateway, Lambda, Step
Functions, SQS, EventBridge, Cognito, CloudFront, Amplify, VPC/NAT. No language
model participates in any decision.

---

## 1. Request path

```mermaid
flowchart TD
    J["Judge / user browser"]
    AR["AWS App Runner<br/>agrivision-inspection · 1 vCPU / 2 GB"]
    API["FastAPI<br/>same-origin static UI + JSON API"]
    ORC["Bounded Agentic Vision orchestrator<br/>18 explicit states · closed 15-tool registry"]
    CV["OpenCV 5.0.0 perception<br/>segmentation · ROI quality · artefacts"]
    DNN["OpenCV DNN (cv2.dnn)<br/>MobileNetV3-Large, ONNX"]

    S3["Amazon S3<br/>model.onnx + manifest.json<br/>private · versioned · encrypted"]
    DDB["Amazon DynamoDB<br/>inspection traces · 14-day TTL"]
    CW["Amazon CloudWatch<br/>structured JSON logs · metrics"]
    ECR["Amazon ECR<br/>container image, pinned by digest"]

    J -->|HTTPS| AR
    AR --> API
    API --> ORC
    ORC --> CV
    ORC -->|only when the capture passes the gate| DNN

    S3 -.->|fetched and SHA-256 verified at start-up| DNN
    ECR -.->|image pulled at deploy| AR
    ORC -.->|trace written, failure is soft| DDB
    API -.->|every step, redacted| CW
    API -->|trace read back by run id| DDB
```

### Why App Runner and not something larger

The workload is one stateless container that must answer an HTTP request in
under a second. App Runner provides TLS, a public URL, health checks, rolling
deployment and autoscaling with no VPC, no load balancer and no orchestrator to
operate. A Lambda would have to cold-start OpenCV and a 12 MB ONNX graph on an
unpredictable schedule; ECS or EKS would add a control plane nobody is going to
maintain after judging. The cost of the whole account is roughly **$0.72/month**
outside of judging traffic.

---

## 2. Identities

```mermaid
flowchart LR
    DEP["agrivision-deployer<br/>IAM user, scoped"]
    AUD["agrivision-auditor<br/>IAM user, read-only"]
    RUN["AgriVisionAppRunnerInstanceRole<br/>assumed by the container"]

    DEP -->|ECR push · App Runner deploy| ECR2["ECR / App Runner"]
    AUD -->|logs:FilterLogEvents only| CW2["CloudWatch logs"]
    RUN -->|s3:GetObject on one prefix| S32["S3 model artifact"]
    RUN -->|PutItem / GetItem on one table| DDB2["DynamoDB traces"]
    RUN -->|PutLogEvents| CW2
```

Three identities, three jobs, and no overlap:

| Identity | Can | Cannot |
| --- | --- | --- |
| `agrivision-deployer` | push images, deploy the service, read `/aws/apprunner/*` logs | read IAM, list buckets, touch DynamoDB, list log groups account-wide |
| `agrivision-auditor` | read the application log group | everything else — App Runner, S3, IAM, DynamoDB, ECR, and every write |
| `AgriVisionAppRunnerInstanceRole` | read the model object, write and read traces, write logs | anything outside those three resources |

The auditor exists for **separation of duties**, not because the capability was
missing: the deployer has held scoped `/aws/apprunner/*` log read since Phase 4.
Phase 6 recorded the opposite, inferring it from a `DescribeLogGroups` probe —
an account-wide call needing `Resource: "*"`, denied by a resource-scoped policy
— and generalising without testing. The Phase 7 record corrects it.

**The container holds no long-lived AWS credentials.** It assumes the instance
role at runtime. Root has **zero** persistent access keys and MFA enabled, and
root is not required for deployment.

---

## 3. Start-up, and what makes the service refuse to serve

```mermaid
sequenceDiagram
    participant AR as App Runner
    participant API as FastAPI
    participant S3 as S3
    participant M as cv2.dnn

    AR->>API: start container
    API-->>AR: port opens immediately
    Note over API: initialisation runs on a background thread,<br/>so /health can answer while the model loads
    API->>S3: GET model.onnx + manifest.json
    S3-->>API: bytes
    API->>API: SHA-256 vs EXPECTED_MODEL_SHA256
    alt digest matches
        API->>M: load graph
        API-->>AR: /ready 200 ready=true
    else digest differs or object missing
        API-->>AR: /ready 503 ready=false
        Note over API: FAIL CLOSED — the service never<br/>infers with an unverified model
    end
```

`/health` is liveness and `/ready` is *can actually inspect*. They are genuinely
different: a clean-clone rehearsal observed `/health` 200 alive alongside
`/ready` false with `model_artifact_present: false`, which is the intended
behaviour and not a degraded mode.

---

## 4. Data, and what is never stored

| Data | Where | Retention |
| --- | --- | --- |
| Uploaded image bytes | held in memory for the request | **never persisted** (`retain_uploads` false) |
| Inspection trace | DynamoDB | 14-day TTL |
| Structured step logs | CloudWatch | account default |
| Model artifact | S3, private and versioned | permanent, digest-pinned |

Logs are scrubbed before emission. A CloudWatch audit of 400 live records found
no credential, authorization header, private key, local filesystem path, S3 URI
or embedded image blob.
