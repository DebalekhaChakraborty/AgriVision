# AgriVision — Technical Report

**OpenCV AI Competition 2026 (powered by AWS)** · Overall + Agentic Vision Award
Branch `competition/opencv-aws-2026` · Live: <https://yp2ajauzkm.us-east-1.awsapprunner.com>

---

## Result labels used throughout

Four kinds of number appear in this report, and they are never mixed. A figure
without one of these labels should be treated as an error.

| Label | Meaning |
| --- | --- |
| **DEVELOPMENT** | measured while the thing being measured was still being built; used to make decisions, not to make claims |
| **CONFIRMATORY** | measured once, on fresh data, against a system frozen beforehand and expectations fixed beforehand |
| **EXPLORATORY** | a diagnostic run after seeing a result, to explain it; never a replacement for it |
| **RETROSPECTIVE ABLATION** | a comparison defined after the results were known; informative, not evidence of the same standing |

A Phase 3 scenario result on drawn fixtures and a Phase 6 result on photographs
are different populations. They do not average.

---

## 1. Executive summary

AgriVision inspects a photograph of a single piece of produce and decides **what
to do next**. That decision — correct the capture, ask for a new one, escalate
to a person, or proceed to the condition model — is made by a deterministic
policy reading OpenCV 5 measurements, and every decision records the metric, the
threshold it crossed and how much that evidence is trusted.

The claim we make, and can support:

> On fresh, licence-verified real photographs the deployed AWS service selects a
> different next action when the OpenCV 5 evidence changes, never runs the
> condition model on a capture it has blocked, and records for every decision
> the metric and threshold that produced it.

Measured on the deployed service, against a system frozen before the data was
opened (**CONFIRMATORY**):

- **10 of 12** real base photographs change their selected action when only the
  controlled visual condition changes
- **58/58** decisions carry the evidence they rest on
- **0/86** runs invoked the model on a capture the agent had blocked
- **12/12** severely blurred photographs produced a recapture request and no
  inference

No language model participates in any decision. Actions are enum members from a
closed 15-tool vocabulary; no free-form string can execute a tool.

---

## 2. Problem and target user

Someone holding a phone in a market, a kitchen or a smallholding wants to know
whether produce looks deteriorated. The hard part is not classification — it is
that most real captures are bad. They are dim, blurred, backlit, or contain six
oranges instead of one.

A classifier answers every one of them with equal confidence. That is the
failure this project is about: a system that cannot tell "this looks rotten"
from "I cannot see this properly" will confidently report on pixels that carry
no information.

The target user is a person, not a pipeline, so the useful output is often an
*instruction* — take it again, move the light — rather than a label.

---

## 3. Scope and claim boundary

**Deployment input contract: one primary produce item per inspection capture.**
Market stalls, piles, crates and trees carrying multiple fruits are out of
scope and return a recapture or human-review action. This is a scope boundary,
not a food-safety statement.

AgriVision reports **visible surface condition only**. It does not detect
pathogens, toxins, microbiological contamination or internal spoilage, and it
does not determine whether food is safe to eat. A person remains responsible
for any decision about produce. This vocabulary is absent from the schemas by
design and asserted absent by test.

Three fruit types: apple, banana, orange.

---

## 4. System architecture

Full diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).

Browser → App Runner → FastAPI → 18-state bounded orchestrator → OpenCV 5
perception → `cv2.dnn` MobileNetV3. Supporting: S3 (model artifact, SHA-256
verified), DynamoDB (traces, 14-day TTL), CloudWatch (structured logs), ECR
(image), IAM (three separated identities).

No Bedrock, no SageMaker, no API Gateway, no Lambda, no VPC. The container holds
no long-lived AWS credentials.

---

## 5. Why OpenCV 5 is substantive here

OpenCV is not a decoding convenience in this system. It produces the evidence
that control decisions are made from.

- **Segmentation** — saturation/Otsu with morphological cleanup and geometric
  guards, plus a classical fallback ladder (chroma distance, GrabCut) that ships
  **disabled**.
- **ROI-restricted quality** — every quality metric is measured on the subject,
  not the frame. Phase 2b found ROI Laplacian variance is about 5% of the
  whole-image value, so whole-image thresholds were measuring the backdrop.
- **The blur metric was chosen, not assumed.** Four candidates were measured
  against a rule written down first, including an invariance criterion that
  could disqualify a metric however well it separated classes. Laplacian
  variance separates blurred from sharp perfectly (AUC 1.000) and its median
  moves **88-fold** across an exposure ladder on the same photographs — so no
  global threshold on it can exist. `high_frequency_ratio` reaches AUC 0.999 at
  1.13×/1.10×/1.03× invariance and is what the policy gates on.
- **Artefact detection** — clipping, glare, occlusion geometry.
- **Remediation** — CLAHE and gamma correction, applied and then *re-measured*.
- **Inference** — `cv2.dnn` executes the ONNX graph. 200/200 class agreement
  with `onnxruntime`, mean max logit difference 2.9e-06.

A resolution bug found in Phase 2c-B is the clearest illustration that these are
real measurements: a fixed 7 px cleanup kernel is 2.7% of a 256 px fixture and
0.36% of a 1920 px photograph, so the fragmentation guard had become a
resolution detector and rejected 71% of samples. Made scale-relative at 7/256,
clean-base mask validity went **42.9% → 85.7%** with 256 px behaviour unchanged.

---

## 6. Agentic Vision design

An 18-state machine with a declared transition table. Illegal transitions raise.
Five terminal states. `INSUFFICIENT_VISUAL_EVIDENCE` is a waypoint, not a
terminal — the run still has to decide which human action to request.

**Evidence maturity** is enforced in code:

| Maturity | May gate a decision? |
| --- | --- |
| CALIBRATED | yes |
| PROVISIONAL | yes, recorded as provisional |
| ADVISORY | only when a policy explicitly permits it, and that permission is recorded |
| UNQUALIFIED | never |

A blocking decision supported only by ADVISORY or UNQUALIFIED evidence raises
`EvidenceMaturityError`. Glare is ADVISORY (0.500 detection on unseen groups,
0.167 false positives — below the 0.70 floor that would let it block). Subject
visibility is UNQUALIFIED and never gates.

**Bounds:** 1 remediation, 1 inference, 2 segmentations, 24 steps. The remediation
cap is structural — the state machine has no edge permitting a second attempt.

Full workflow: [AGENT_WORKFLOW_DIAGRAM.md](AGENT_WORKFLOW_DIAGRAM.md).

---

## 7. Visual evidence

The primary evidence is the confirmatory counterfactual result in §13: **10 of
12** fresh real bases change their selected action when only the controlled
visual condition changes.

### An illustrative example — one real photograph, four actions

**This is one base, chosen to be legible, not because it is representative.**
It is the strongest of the twelve; §13 gives the aggregate and §15 gives the
failures. All four rows come from the deployed service under identical policy
fingerprints, and differ only in pixels.

| Condition | OpenCV 5 evidence | Agent action | Next tool | Model? |
| --- | --- | --- | --- | --- |
| Reference | `high_frequency_ratio` 0.578, `mean_luminance` 0.479, `contrast` 0.557 | `NONE` | `run_condition_model` | **yes** |
| Underexposed | `mean_luminance` **0.040** — below the calibrated floor 0.0527 | `APPLY_GAMMA` | re-segment, re-measure | yes, after correction |
| Recoverable contrast loss | `contrast` **0.125** — below the advisory limit 0.1314 | `APPLY_CLAHE` | re-segment, re-measure | yes, after correction |
| Severe blur | `high_frequency_ratio` **0.255** — below the calibrated floor 0.3182 | `REQUEST_RECAPTURE` | **none** | **no** |

Read the last row against the first. The luminance and contrast are essentially
unchanged from the reference — 0.484 and 0.529 — so nothing about the exposure
explains the outcome. The single metric that moved is high-frequency content,
and it moved across a threshold that was calibrated before this photograph
existed. The agent did not merely decline: it declined **for a stated reason**,
and it did not invoke the model.

That is the whole claim in one subject: the OpenCV measurement is the only thing
that changed, and it is what changed what happened next.

**Mechanism support (DEVELOPMENT):** the Phase 3 constructed-fixture
counterfactual established the same behaviour deterministically on drawn
subjects, where every pixel is controlled. It is supporting evidence for the
mechanism, not evidence about photographs, and the two are not pooled.

---

## 8. Foreground and quality pipeline

Primary segmentation is saturation/Otsu. Masks pass geometric guards that detect
the failure shapes classical segmenters actually produce: grabbing the whole
frame, grabbing nothing, grabbing the background, or shattering into fragments.
These do not measure accuracy — no ground truth exists on real photographs.

A Phase 3b fallback ladder (chroma distance, downscaled GrabCut) recovers
additional masks. It is **PROVISIONAL and disabled by default**, because every
Phase 2c-B, 2d and 3 result was produced with the primary alone and a default
that silently changed which mask those numbers describe would invalidate them
rather than improve them.

**CONFIRMATORY, Phase 6:** foreground-valid **29/38** on fresh natural
photographs obeying the input contract.

---

## 9. Condition model and `cv2.dnn`

**MobileNetV3-Large (V2 Experiment 004) → ONNX → `cv2.dnn`.** 11.6 MB.

**This was an engineering trade-off, not a claim that it is the best model.**
Stated plainly: SigLIP2 strict zero-shot holds **86.3%** on its worst external
domain against the selected model's **70.4%** — the selected model is
**15.9 points worse** on unfamiliar produce photography. DINOv2 (330 MB encoder)
and CLIP (571 MB encoder) were also stronger and also rejected.

What was bought: an 11.6 MB artifact that runs through `cv2.dnn` with no torch,
no transformers and no onnxruntime in the serving image, at a **22 ms median**
inference. The whole container is 719 MB rather than multiple gigabytes.

The trade-off is defensible precisely *because* of the agent: a model more
likely to be wrong on unfamiliar photography is safer behind a gate that
declines to invoke it on captures that cannot support a result.

**Model confidence is UNQUALIFIED for policy.** No confidence threshold exists
anywhere in the system and none was introduced. Confidence is reported
descriptively as domain-shift evidence.

---

## 10. Bounded decision loop

Covered in §6. The property worth repeating: the loop's bounds are structural
rather than counted, and **BOUNDED_EXECUTION_RATE was 48/48** on the
confirmatory scenarios.

---

## 11. AWS deployment

One container on App Runner. The model artifact is fetched from S3 and its
SHA-256 verified against a pinned expectation at start-up; a mismatch leaves the
service **not ready** rather than serving with an unverified model. Verified
directly in a clean-clone rehearsal: before the artifact was mounted, `/health`
answered 200 alive while `/ready` answered false.

Traces persist to DynamoDB and a persistence failure is soft — it must never
cause the system to invent an inspection result. Logs are structured JSON,
scrubbed before emission.

Three separated identities (§ARCHITECTURE). Root has zero persistent access keys,
MFA enabled, and is not required for deployment.

---

## 12. Evaluation protocol

The discipline that makes the Phase 6 numbers worth anything:

1. **Freeze the system first.** Fingerprint `97b059be36bc3fbe` over the commit,
   every policy fingerprint, every component version, the model manifest and the
   execution budget — recorded before any Phase 6 image was fetched. Regenerated
   after the evaluation tooling was written and returned **identical**.
2. **Fresh sources.** Independence enforced at three levels: image id, creator,
   and category. Measured overlap with all prior corpora: **0 groups, 0 ids,
   0 content hashes.**
3. **Preregister.** Expected actions fixed at fingerprint `9ce69ebe042428dc`
   before a single variant ran, along with **both** denominators.
4. **Evaluate the deployed service**, not a local pipeline.
5. **Do not retune.** Nothing was changed in response to a result.

Full method: [PHASE6_FINAL_EVALUATION.md](PHASE6_FINAL_EVALUATION.md).

---

## 13. Confirmatory results

### Canonical result table — CONFIRMATORY, deployed service

| Measurement | Result | Denominator |
| --- | --- | --- |
| Fresh natural photographs | **38** | 38 source groups, 8 licences |
| Complete inspection | **28/38 (73.7%)** | all Track R images — **not accuracy** |
| Foreground failure | **9/38 (23.7%)** | all Track R images |
| Recapture requested | 10/38 (26.3%) | all Track R images |
| **Fruit-type classification** | **27/28 (96.4%)** | **model invocations only** |
| Controlled real-image scenarios | **48** | 12 bases × 4 variants |
| First-action agreement | **30/48 (62.5%)** | all scenarios |
| — restricted to bases that segmented | 27/36 (75.0%) | secondary, fixed in advance |
| Task success | **29/48 (60.4%)** | all scenarios |
| **Decision attribution** | **58/58 (100%)** | all recorded decisions |
| **Unsafe inference — Track C** | **0/48** | all scenarios |
| **Unsafe inference — Track R** | **0/38** | all images |
| Remediation success | 9/10 | corrections attempted |
| Remediation harm | 1/10 | corrections attempted |
| Bounded execution | 48/48 | all scenarios |
| Trace completeness | 48/48 | all scenarios |
| Fail-safe | 12/12 | `SEVERE_BLUR` scenarios |

**The fruit-type denominator is 28, not 38.** Ten images were blocked before the
model ran; they are neither right nor wrong. Restating 27 over 38 would convert
a model measurement into an end-to-end one, which nothing here supports. A test
enforces this.

### Per-variant first-action agreement

| Variant | Agreement |
| --- | --- |
| `SEVERE_BLUR` | **12/12** |
| `REFERENCE` | 8/12 |
| `LOW_CONTRAST_RECOVERABLE` | 8/12 |
| `UNDEREXPOSED_RECOVERABLE` | **2/12** — see §15.1 |

### Counterfactual evidence

**10 of 12** real bases change their selected action when only the controlled
visual condition changes, under identical policy fingerprints. Four distinct
first actions observed: `NONE`, `APPLY_CLAHE`, `APPLY_GAMMA`,
`REQUEST_RECAPTURE`. The two invariant bases are those whose foreground could
not be isolated, so every variant stopped at the same earlier refusal; they are
listed, not dropped.

### Deployed latency — client-observed, includes network

| Path | n | Median | p95 |
| --- | --- | --- | --- |
| Track R natural | 38 | 1326 ms | 3114 ms |
| Track C remediation | 10 | 2023 ms | 6265 ms |
| Track C recapture | 29 | 992 ms | 3479 ms |

Not server compute time. Server-side model inference is a 22 ms median of that.

---

## 14. Agent-disabled retrospective ablation

**RETROSPECTIVE ABLATION — defined after the Phase 6 results were seen. Not
preregistered, not confirmatory.**

Baseline: `decode → preprocessing contract → cv2.dnn forward pass`. No gate, no
remediation, no branching. Same model file, same artifact, same preprocessing.
Validity checked rather than assumed: the local baseline reproduced the deployed
predictions **28/28** on shared images.

### Track R

| | Model-only | Agentic |
| --- | --- | --- |
| Inference coverage | 38/38 | 28/38 |
| Fruit-type correct | **37/38** | 27/28 invoked |
| Abstentions | 0 | 10 |
| Recapture capability | no | yes |
| Unsafe-inference guard | no | yes |

**The agentic gate did not improve fruit-type classification on this
fresh-real-image sample.** That is the finding, stated without softening.

The baseline classified **10 out of 10** of the images the agent declined
correctly, and the single fruit-type error in the whole sample occurred on an
image the agent *did* pass through. On this question the gate cost ten images of
coverage and prevented no error.

The honest reading is that fruit type is the wrong question for the gate to be
judged on. Fruit type is shape and colour, which survive blur and underexposure;
visible condition is fine surface texture, which is exactly what blur destroys.
Fruit type is the only independently labelled quantity available, so it is what
can be measured — and it does not support the gate. **The gate's value on the
task the product actually claims remains unmeasured**, because no
visible-condition label exists for real photographs.

### Track C — what a classifier structurally cannot do

| Metric | Baseline | Agentic |
| --- | --- | --- |
| `POLICY_BYPASS_RATE` | **36/36** | — |
| `UNSAFE_INFERENCE` (blocking scenarios) | **12/12** | **0/12** |
| `UNCORRECTED_RECOVERABLE` | 24/24 | — |

`POLICY_BYPASS_RATE` is the fraction of scenarios whose preregistered action is a
correction or a refusal where the baseline would infer anyway; the denominator
excludes the 12 `REFERENCE` scenarios, where proceeding is correct.
`UNSAFE_INFERENCE` is reserved for scenarios where inference must be blocked
outright — a recoverable condition left uncorrected is counted separately and is
not called unsafe.

On severely blurred photographs the baseline produces a confident-looking
condition answer from pixels that no longer contain surface detail, 12 times out
of 12. The agent does so zero times out of 12.

**This is a policy and safety-control contribution. It is not evidence that the
agent improves classification accuracy** — §14's Track R result is evidence
against that, on the one labelled quantity available. The two findings sit side
by side because both are true.

---

## 15. Failure analysis

Phase 6 taxonomy across both tracks:

| Category | Count |
| --- | --- |
| `FOREGROUND_FAILURE` | 19 |
| `SCENARIO_CONSTRUCTION_DEFECT` | 7 |
| `MODEL_FRUIT_TYPE_ERROR` | 1 |
| `POLICY_MISMATCH` | 1 |
| `SOURCE_IMAGE_BLOCKING_CONDITION` | 1 |

### 15.1 The underexposure route scored 2/12 — CONFIRMATORY

This is the weakest confirmatory result in the project and it is stated here,
not in an appendix.

The preregistered expectation was `APPLY_GAMMA` on a recoverable
underexposure. Agreement was **2/12**. Measuring where the variants' pixels
actually landed against the calibrated thresholds:

| Landing zone | Bases |
| --- | --- |
| Shadows clipped — detail destroyed | 7 |
| Above the underexposure floor — nothing to correct | 2 |
| No valid foreground | 3 |
| **Actually in the recoverable band** | **0** |

**Not one of the twelve variants occupied the state the scenario was named
after.** The exposure gain of 0.12 came from synthetic development evidence — a
256×256 fixture with a mid-grey background. Real photographs already contain
deep shadows, so the same gain drives them through the narrow recoverable band
and into clipping.

The agent was reading the pixels correctly and refusing images whose shadow
detail had genuinely been destroyed. The scenario was mislabelled by its own
construction, which is why it is classified `SCENARIO_CONSTRUCTION_DEFECT` and
not `QUALITY_FALSE_BLOCK`.

**No post-confirmatory retuning was performed.** The variant was not rebuilt and
rerun as confirmatory; adjusting evaluation data until it yields the
preregistered answer is precisely what preregistration prevents.

**EXPLORATORY** follow-up, reported separately and never substituted for the
2/12: sweeping the exposure gain per base found the recoverable band is
reachable on only **3 of 12** real photographs — for the other nine, any gain
low enough to cross the floor (ROI mean luminance 0.0527) clips the shadows
first. On all **3/3** where it is reachable, the agent selected `APPLY_GAMMA`.

The route works. The calibrated band is narrow and hard to reach by global
darkening of a real photograph. **Future work should calibrate adaptive
degradation and remediation behaviour on real-device captures**, which is a
recalibration exercise on new data, not a tuning of the current policy.

### 15.2 Segmentation is the binding limit

19 of 29 classified failures are foreground failures. Reasons on Track R:
`EXCESSIVE_BORDER_CONTACT` 6, `MASK_FRAGMENTED` 2, `NO_DOMINANT_COMPONENT` 2.

### 15.3 The other three

- **`SOURCE_IMAGE_BLOCKING_CONDITION`** — one unmodified reference photograph
  carries clipped shadows of its own. The agent refused correctly; the
  preregistration assumed every fresh photograph would be usable.
- **`POLICY_MISMATCH`** — one low-contrast variant landed just above the
  advisory limit, so the agent did nothing and completed.
- **`MODEL_FRUIT_TYPE_ERROR`** — an apple predicted banana at confidence 0.233,
  the lowest-confidence prediction in the set.

### 15.4 A controlled variant changed segmentation validity

On base `WC-41881957d489f4` the `REFERENCE` foreground was valid but the
`LOW_CONTRAST` variant lost it. Recorded as `FOREGROUND_FAILURE` and counted as
a scenario failure. The expectation was not rewritten to accommodate it.

---

## 16. Responsible use

Reproduced verbatim from the live page:

> AgriVision assesses **visible produce condition** from images. It does not
> detect pathogens, toxins, microbiological contamination or internal spoilage,
> and it does not determine whether food is safe to eat. A person remains
> responsible for any decision about produce.

The UI subordinates the classifier prediction to the agent outcome, shows
evidence maturity so advisory glare cannot be read as calibrated, describes
remediation as re-measurement after a correction rather than as restored
information, and presents recapture and human review as guidance rather than
errors. Counterfactual variants are labelled a **controlled demonstration** in
the API response itself, not only in the page.

---

## 17. Security and privacy

Audited in Phase 7; full record in `results/phase7/security_audit.json`.

- **667 tracked files and 31 commits scanned** for credential shapes — zero findings
- No `.env`, private key, or credential file tracked
- Committed images carry no camera or location EXIF (they are generated plots)
- Model bucket: all four S3 public-access blocks on
- Root: **zero** persistent access keys, MFA enabled, not required for deployment
- Container holds no long-lived AWS credentials; it assumes an instance role
- Upload limit 12 MB, magic-byte content validation (not the declared
  Content-Type or the filename), model checksum verified fail-closed
- Uploaded images are never persisted

A CloudWatch audit of 400 live records found no credential, authorization
header, private key, local path, S3 URI or embedded image blob.

An AWS account id inside an ARN is an infrastructure identifier, not a secret,
and is deliberately not redacted; over-redacting harmless identifiers makes a
security report harder to check without making anything safer.

---

## 18. Reproducibility

A clean-clone rehearsal was performed in a new directory from the public remote,
with a fresh virtualenv and no reuse of any existing environment, cache or
generated file. **All 13 documented steps passed**, and the clean clone
reproduced the deployed policy fingerprint `78b1e2151773787a` exactly.

**Three defects were found by the rehearsal and fixed**, all of which had been
masked by a working development environment:

1. The test HTTP transport was unpinned, so collection failed on two modules in
   any genuinely fresh install. Pinned to the version every recorded result was
   produced under.
2. A model-dependent test failed rather than skipping without the ONNX file.
3. A gitignore assertion queried a directory pattern without its trailing slash,
   which passes where the directory exists and fails in a fresh clone.

**Test classification:**

| Mode | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| Artifact-independent (no ONNX file) | 770 | 141 | 0 |
| With the model artifact | 909 | 2 | 0 |

The two remaining skips are the research checkpoint and the licensed corpus
imagery, neither of which is in the repository.

### 18.1 The model artifact, and what can be reproduced without it

The ONNX artifact is **intentionally not published**. Stating the mechanism
plainly so the reproduction claim is neither overstated nor mysterious:

| | |
| --- | --- |
| Artifact | `competition/models/artifacts/mobilenetv3_large_v2exp004/` |
| Contents | `model.onnx` (11,906,184 bytes) **and** `manifest.json` (both required) |
| SHA-256 | `77d8614671873cb41ebfd725f0a4b468beca9c30cfe6a3361217343917324d4b` |
| Runtime loading | `cv2.dnn.readNetFromONNX` via `load_condition_model`, which also reads the manifest |
| Deployment mechanism | fetched from a private, versioned, encrypted S3 bucket at container start-up, then SHA-256 verified against `EXPECTED_MODEL_SHA256` before the graph is loaded |
| On mismatch or absence | the service stays **not ready**; it never infers with an unverified model |

**The manifest is not optional.** It carries the class ordering, the
preprocessing contract and an `onnx_sha256_prefix` that is checked against the
weights, so loading the graph without it would mean guessing the label order and
the normalisation. Fetching only `model.onnx` produced a verified-but-unloadable
model during Phase 4, which is how this became explicit.

**Why it is excluded from the public repository:** it is a build product derived
from a V2 research checkpoint, and this project does not redistribute training
data or model weights. Committing an 11.6 MB binary to make a reproduction claim
look stronger would be publishing someone else's derived artifact for
presentation convenience.

**What that means for a third party, stated without hedging:**

- **Artifact-independent reproduction is public and complete.** Anyone can clone
  the repository and run 770 tests covering the perception pipeline, the policy
  engine, the state machine, the evaluation discipline and the service
  contracts. Model-dependent tests skip explicitly rather than failing.
- **The live judge-accessible endpoint demonstrates the complete deployed
  system**, model included, with no credentials required of the judge.
- **Model-dependent *local* reproduction additionally requires the external
  artifact**, which means access to the private S3 object or to the V2
  checkpoint the export step reads. A third party cloning the public repository
  cannot obtain it unaided.

This is a real limitation and it is not worked around by publishing the weights.

---

## 19. Known limitations

- **Segmentation is the binding limit** — 76.3% foreground-valid under a contract
  that already excludes the hard cases, and it would not survive market stalls.
- **The recoverable-underexposure band is narrow** and reachable on only 3 of 12
  real photographs by global darkening (§15.1).
- **Visible-condition accuracy on real imagery is unmeasured** — no independent
  label exists, and the model's own output cannot be used as its own ground
  truth. This is the largest gap in the evaluation.
- **The gate's benefit is unproven on the task it exists for.** On fruit type it
  cost coverage and prevented no error (§14).
- **Small samples** — 38 photographs and 48 scenarios. No confidence intervals
  are drawn because at these counts they would be wide enough to be useless and
  decorative enough to look authoritative.
- **Orange is under-represented** at 8 images against 15 apple and 15 banana.
- **Wikimedia Commons is not the deployment domain.** It is licence-verifiable
  real photography, which is a different and lesser thing than phone captures.
- **No phone-camera validation** has been collected.
- **The selected model is 15.9 points worse** than the best candidate on its
  worst external domain (§9).
- **Glare is advisory only** — 0.500 detection on unseen groups with 0.167 false
  positives. Subject visibility is unqualified and never gates.
- **Single-rater curation** for Track R membership.
- **The endpoint is unauthenticated**, which is acceptable for a judged demo.

---

## 20. Future work

In the order the evidence argues for:

1. **Collect real-device captures** and calibrate adaptive degradation and
   remediation on them. This addresses §15.1 directly and is a recalibration on
   new data, not a retune of the current policy.
2. **Obtain visible-condition labels** for real photographs so the system's
   actual claim can be measured rather than proxied by fruit type.
3. **Improve segmentation**, which gates everything downstream. The Phase 3b
   fallback exists and is measured; enabling it is a deployment decision whose
   consequences are documented.
4. **Widen the input contract** beyond one primary item, which is the boundary
   most likely to frustrate a real user.
5. **Publish a credential-free route to the model artifact** so the container is
   reproducible by a third party.
