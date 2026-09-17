# AgriVision — OpenCV AI Competition 2026 Technical Blueprint

**Agentic Visual Intelligence for Produce Quality Inspection**

This document is the single source of truth for the competition project. It is
scoped to the `competition/opencv-aws-2026` branch and describes planned work.
It contains no experimental results of its own. Every number quoted from prior
research is attributed to the experiment that produced it.

| Field | Value |
| --- | --- |
| Competition | OpenCV AI Competition 2026, powered by AWS |
| Award targets | Overall Competition; Agentic Vision Award |
| Out of initial scope | COOL award |
| Branch | `competition/opencv-aws-2026` |
| Competition base commit | `3928d43b3bdd3a754f98f1f411596050de29da17` |
| Base commit subject | Complete AgriVision V2 research experiments through Experiment 015 |
| Research lineage | `legacy` `9769e3c` (frozen V1) → `master` `3928d43` (V2 research) |
| Status | Phase 0 — architecture and foundation only; no implementation |

---

## 1. Project vision

Produce quality inspection today is manual, subjective and inconsistent. A
single inspector judging a crate of apples applies different standards at
hour one and hour eight, and no record survives the decision.

AgriVision is an **agentic visual inspection system** for produce. It does not
merely classify an image. It evaluates whether the image it was given is fit to
be judged at all, acts to improve or re-acquire the evidence when it is not,
localises the visible surface regions driving its assessment, and escalates to a
human when its own evidence does not support a confident, safe conclusion. Every
step is recorded as an inspectable trace.

The distinguishing claim is that **OpenCV 5 measurements change what the system
does next** — not merely what it reports.

## 2. Competition objective

Deliver an entry that satisfies every mandatory submission requirement, competes
credibly for the Overall Competition, and demonstrably qualifies for the Agentic
Vision Award by proving that OpenCV 5 visual evidence influences subsequent
decisions, tool calls, re-analysis and human-approval requests.

Judging weights this project is optimised against:

| Criterion | Weight | Primary response |
| --- | --- | --- |
| Technical execution | 30% | Substantive OpenCV 5 pipeline, real agent loop, tests, measured evaluation |
| Innovation | 20% | Perception-driven control, not a classifier with a chat wrapper |
| Real-world impact | 20% | Agriculture domain, deployable inspection workflow, honest limitations |
| UX | 10% | Judge-operable demo showing evidence and trace, not just a label |
| Documentation / presentation | 10% | This blueprint, technical report, architecture diagram, video |
| AWS / reproducibility / responsible operation | 10% | Pinned deps, IaC, least-privilege IAM, observability, claim boundary |

## 3. Current repository baseline

The repository carries three lines. The competition branch is additive to all of
them and modifies none.

| Line | Commit | Content |
| --- | --- | --- |
| `legacy` | `9769e3c` | Frozen 2018 V1 artifact. Never modified. |
| `master` | `3928d43` | V2 research programme, Experiments 001–015. |
| `competition/opencv-aws-2026` | from `3928d43` | Competition work only. |

Verified environment facts as of Phase 0:

- `opencv-python==5.0.0.93` is installed in `.venv-v2` and imports as `cv2 5.0.0`.
- `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, `numpy 2.4.6`, `Pillow 12.3.0`.
- The repository contains **no tests, no CI, and no packaging configuration**.
- OpenCV is currently unused by any module in `src/` or `v2/src/`; it appears
  only in the two requirements files and one V1 notebook.

## 4. Historical V1 assets available for reuse

V1 (`legacy`, and the frozen tree on `master`) provides:

- `model/fruit_freshness_cnn.h5` — trained Keras classifier.
- `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/dataset_split.py`.
- `app.py` — a small Flask upload-and-predict demonstration.
- `results/*.png` — V1 training and evaluation figures.
- `DATASET_PROVENANCE.md`, `TEMPORAL_AUDIT.md` — provenance discipline worth
  carrying into the competition entry.

**Reuse boundary.** V1 pins `tensorflow==1.12.0` / `Keras==2.2.4` on CPython
3.6.7. V2 and the competition line run PyTorch on Python 3.11. These cannot
coexist in one process or one container. V1 is therefore reused as a **reported
historical baseline number and as UX precedent**, never as a runtime dependency
of the competition system.

## 5. V2 research boundary

The competition line may **read** V2 research outputs and **import** V2 modules
for inference. It must not:

- alter any file under `v2/results/`, `v2/experiments/`, or the analysis docs;
- retrain or re-run research experiments to produce competition numbers;
- publish competition measurements as V2 research results, or the reverse;
- be merged wholesale back into `master`.

Scientific improvements discovered during competition work may be selectively
cherry-picked back to `master` after review, as isolated commits.

## 6. Competition scope

In scope:

1. An OpenCV 5 perception layer producing quantified visual evidence.
2. A condition model assessing visible produce state.
3. An agentic perception–decision–action loop with a recorded trace.
4. An AWS deployment exposing a working endpoint.
5. A judge-operable demo UI surfacing evidence and trace.
6. An evaluation suite with baselines, failure taxonomy and latency.
7. Technical report, architecture diagram, video, submission materials.

## 7. Explicit non-goals

- Not a food-safety, pathogen, toxin, contamination or edibility system.
- Not an internal-spoilage or shelf-life predictor.
- Not a retraining platform; no new research claims.
- Not a multi-tenant production service; no authentication/billing/accounts.
- Not a mobile application.
- No COOL award submission in initial scope.
- No microservice decomposition, message buses, or orchestration services that
  the workload does not require.

## 8. Target users

| User | Need | System response |
| --- | --- | --- |
| Intake inspector at a packhouse | Fast, consistent, recorded judgement | Inspection with evidence and trace |
| Quality supervisor | Review escalated or low-confidence cases | Human-review queue with localised evidence |
| Smallholder / cooperative agent | Guidance with a phone camera in poor light | Recapture and enhancement guidance |
| Auditor | Evidence that a decision was justified | Immutable per-inspection trace record |

## 9. User journey

1. User submits an image or short video frame of produce.
2. Perception measures capture quality and localises candidate regions.
3. If capture quality is inadequate, the agent enhances and re-analyses, or asks
   for a recapture with a specific, actionable reason.
4. The condition model assesses visible state; the agent cross-checks the model
   against perception evidence.
5. On conflict or low confidence, the agent re-analyses the suspicious region or
   escalates for human review.
6. The user receives a decision, the localised visual evidence, the confidence
   and limitation statement, and the full trace.

## 10. Real-world problem statement

Post-harvest loss is a material agricultural problem, and visual quality
assessment is the first control point where it can be caught. Manual inspection
does not scale, is not reproducible, and leaves no audit trail. Naive automation
is worse than no automation when it is confidently wrong on an image it should
have refused to judge.

This project's own prior research quantifies exactly that danger. From
Experiment 014 (multi-domain robustness, `v2/results/experiment_014_multi_domain/`),
accuracy on the source test domain versus the worst external domain:

| System | Source | Worst external |
| --- | --- | --- |
| SigLIP2 strict P1 zero-shot | 97.9% | 86.3% |
| SigLIP2 + linear probe | 98.6% | 73.5% |
| MobileNetV3-Large | 93.0% | 70.4% |
| ResNet50 | 94.2% | 69.3% |
| EfficientNet-B0 | 94.6% | 68.8% |
| CLIP strict P1 zero-shot | 90.0% | 67.4% |
| DINOv2 + linear probe | 89.1% | 66.4% |
| CLIP + linear probe | 95.8% | 55.2% |
| Custom CNN | 74.1% | 34.6% |

Every system degrades out of domain; the weakest collapses by nearly 40 points.
A deployed inspector will routinely photograph produce under conditions unlike
the training distribution. **This is the evidence base for the entire design: the
classifier alone cannot be trusted, so the system must measure its own input and
act on that measurement.**

## 11. Why OpenCV 5 is technically necessary

OpenCV is not used here to resize images before inference. It is the instrument
that makes the system safe, for four reasons:

1. **Classifier confidence is not a capture-quality signal.** A softmax is
   famously overconfident on out-of-distribution input, which is precisely the
   regime Section 10 documents. Blur, exposure, glare and occlusion must be
   measured *independently of the model* to be trustworthy. OpenCV provides
   deterministic, interpretable, model-free measurements of exactly these.
2. **Localisation is a segmentation and morphology problem.** "Where is the
   deterioration?" is answered by foreground extraction, colour/texture anomaly
   masks and connected-component analysis — native OpenCV work, and explainable
   in a way a saliency heatmap is not.
3. **Remediation requires image operations, not inference.** CLAHE, gamma
   correction, denoising and re-segmentation with different parameters are the
   *actions* the agent takes. Without OpenCV there is no action space.
4. **Cost and latency.** Perception gating runs in milliseconds on CPU and can
   reject or repair an input before any model is invoked. Measured in Phase 1:
   median 3.41 ms, p95 3.66 ms on 256×256 locally.

The perception layer is therefore load-bearing: remove it and the agent has
nothing to decide on and nothing to do.

**Phase 1 added a fifth reason, discovered by measurement rather than assumed.**
The capture-quality metrics are *coupled*: with no blur applied at all,
Laplacian variance scales with roughly the square of exposure gain, so a dark
capture is falsely flagged as blurred (observed at gain ≤ 0.25) and a bright one
can mask real blur. A single-pass assessment is therefore insufficient on its
own terms — exposure must be corrected and sharpness re-measured. The
`assess → enhance → re-assess` loop is a measured necessity, not a design
preference. See [PHASE1_OPENCV_PERCEPTION.md](PHASE1_OPENCV_PERCEPTION.md) §9.1.

## 12. Planned OpenCV 5 pipeline

All operations below are confirmed available in the installed `cv2 5.0.0`
build. `cv2.ximgproc` is **not** available (contrib is not installed), so the
design deliberately uses core modules only.

Stages marked **[P1]** are implemented and tested as of Phase 1; the rest are
planned.

| Stage | Technique | Output |
| --- | --- | --- |
| Decode / normalise **[P1]** | `imread`, `cvtColor` | working image, colour spaces |
| Blur assessment **[P1]** | variance of `Laplacian` | sharpness score |
| Illumination assessment **[P1]** | Lab L* statistics, clipping fractions | exposure score, under/over flags |
| Glare / specular | near-saturation mask, morphology | glare fraction |
| Enhancement (action) | `createCLAHE`, gamma LUT, `fastNlMeansDenoising` | repaired image |
| Foreground / ROI | `grabCut`, `watershed`, `findContours` | produce mask |
| Segmentation quality | solidity, area fraction, `Canny` edge agreement | segmentation confidence |
| Colour analysis | HSV/Lab histograms, brown/dark-spot ratio | discolouration metrics |
| Texture / edge | Laplacian energy, `Sobel`, edge density | surface texture metrics |
| Anomaly localisation | threshold + `morphologyEx` + `connectedComponentsWithStats` | scored candidate regions |
| Evidence rendering | `drawContours`, `rectangle`, overlays | annotated evidence image |

## 13. Perception outputs

A single typed `PerceptionEvidence` record, versioned and trace-serialisable:

- `sharpness`: Laplacian variance, plus normalised score and `is_blurred`
- `exposure`: mean/std luminance, clipped-low and clipped-high fractions,
  `is_underexposed`, `is_overexposed`
- `glare_fraction`, `is_glared`
- `mask`: area fraction, solidity, contour count, `segmentation_confidence`
- `colour`: per-channel histograms, `dark_spot_ratio`, `discolouration_index`
- `texture`: edge density, Laplacian energy
- `regions`: list of `{bbox, area, anomaly_score, dominant_cue}`
- `capture_quality`: aggregate gate score driving control decisions
- `evidence_image_key`: S3 key of the annotated overlay
- `perception_version`: pipeline version for trace reproducibility

## 14. Vision / model architecture

Six classes, unchanged from the research taxonomy: `fresh_apple`,
`fresh_banana`, `fresh_orange`, `rotten_apple`, `rotten_banana`, `rotten_orange`.

Candidate condition models, to be selected in Phase 2 against the criteria
below — accuracy is **not** the sole criterion, out-of-domain retention and
container size matter more for a deployed inspector:

| Candidate | Rationale | Concern |
| --- | --- | --- |
| SigLIP2 strict zero-shot | Best worst-external retention (86.3%, Exp 014) | Encoder size, cold start |
| SigLIP2 / DINOv2 + linear probe | Small probe head over frozen features | Encoder still large |
| MobileNetV3-Large | Smallest deployable footprint | 70.4% worst external |
| V1 Keras CNN | Historical baseline only | Incompatible runtime; not deployed |

The model must expose calibrated uncertainty, not a bare argmax, because the
decision layer consumes uncertainty directly.

## 15. Agentic perception–decision–action loop

```
            ┌──────────────────────────────────────────┐
            │            IMAGE / VIDEO FRAME           │
            └────────────────────┬─────────────────────┘
                                 v
                    ┌─────────────────────────┐
                    │  OpenCV 5 PERCEPTION    │
                    │  quality · ROI · colour │
                    │  texture · anomalies    │
                    └────────────┬────────────┘
                                 v
                    ┌─────────────────────────┐
                    │   CONDITION MODEL       │  (invoked only if
                    │   + uncertainty         │   capture gate passes)
                    └────────────┬────────────┘
                                 v
          ┌──────────────────────────────────────────────┐
          │           AGENTIC DECISION LAYER             │
          │  deterministic policy over OpenCV evidence   │
          │  + model uncertainty  (Bedrock explains)     │
          └───┬────────┬────────┬─────────┬──────────┬───┘
              v        v        v         v          v
          ACCEPT   ENHANCE  RE-SEGMENT  RE-ANALYSE  ESCALATE
                   & RERUN  (new params)  ROI       / RECAPTURE
              │        │        │         │          │
              │        └────────┴─────────┘          │
              │        re-enters perception          │
              v                                      v
        RESULT + EVIDENCE + TRACE            HUMAN REVIEW QUEUE
```

The loop is bounded: a maximum remediation depth, and every iteration appended
to the trace with the rule that fired.

## 16. Agent tools

| Tool | Implementation | Triggering evidence |
| --- | --- | --- |
| `assess_capture_quality` | OpenCV | always, first |
| `enhance_image` | OpenCV CLAHE / gamma / denoise | low exposure or contrast |
| `segment_produce` | OpenCV GrabCut / watershed, parameterised | always; re-invoked on low segmentation confidence |
| `localize_surface_anomalies` | OpenCV morphology + components | after successful segmentation |
| `classify_condition` | condition model | capture gate passed |
| `reclassify_region` | crop → model | model/perception conflict |
| `request_recapture` | returns structured, human-readable reason | unrecoverable capture failure |
| `escalate_for_human_review` | DynamoDB queue write | low confidence, conflict, or domain-shift flag |
| `explain_decision` | Bedrock, over the trace | final narration only |

**Bedrock is deliberately not the controller.** Control decisions are made by a
deterministic, threshold-based policy over OpenCV metrics so they are testable,
reproducible and auditable. Bedrock narrates the decision and may propose a
remediation ordering when the policy is genuinely ambiguous. This avoids the
"generic LLM wrapper" failure mode and keeps the agent's behaviour measurable.

## 17. Human-in-the-loop policy

Escalate when any of the following hold:

- capture quality remains below threshold after bounded remediation;
- model uncertainty exceeds the escalation threshold;
- perception and model disagree (anomalous regions present, model says fresh, or
  the inverse);
- the produce type is outside the six-class taxonomy, or no produce is detected;
- a domain-shift indicator fires.

Escalation is a first-class successful outcome, never a failure. The queue entry
carries the annotated evidence image, the evidence record and the full trace.

## 18. Confidence and uncertainty policy

- The system reports a calibrated confidence, never a bare softmax maximum.
- Confidence is reported **jointly with capture quality**; a high-confidence
  result on a poor-quality capture is presented as low overall reliability.
- Thresholds are configuration, not constants in code, and are recorded in the
  trace so a historical decision can be re-derived.
- Calibration is measured (Section 30), not assumed.

## 19. Failure-handling strategy

| Failure | Detection | Response |
| --- | --- | --- |
| Blurred capture | Laplacian variance below threshold | recapture request (blur is not recoverable by enhancement) |
| Under/over exposure | clipping fractions | CLAHE/gamma, re-run, compare |
| Glare | saturation mask | re-segment, mask glare regions, or recapture |
| Weak segmentation | solidity / area out of band | re-segment with alternate parameters |
| No produce detected | mask area near zero | reject with reason |
| Unknown produce type | out-of-taxonomy indicator | escalate, do not guess |
| Model/perception conflict | rule-level cross-check | ROI re-analysis, then escalate |
| Model unavailable | inference error | degrade to perception-only report, escalate |
| Timeout / depth exceeded | loop bound | return best-so-far with explicit caveat |

## 20. Proposed AWS architecture

```
   Judge / user browser
            │  HTTPS
            v
   ┌──────────────────┐        ┌─────────────────────┐
   │  AWS App Runner  │──────▶ │  Amazon ECR         │
   │  container:      │  image │  OpenCV 5 + model   │
   │  API + perception│        └─────────────────────┘
   │  + agent loop    │
   └───┬────┬─────┬───┘
       │    │     │
       │    │     └─────────▶ Amazon Bedrock   (explanation / ambiguous cases)
       │    │
       │    └───────────────▶ Amazon DynamoDB  (inspection + trace records)
       │
       └────────────────────▶ Amazon S3        (uploads, evidence overlays)

   All components ──▶ Amazon CloudWatch (logs, metrics, alarms)
   All access via scoped IAM task roles
```

**Primary recommendation: AWS App Runner.** It takes a container image directly
to an HTTPS endpoint with no VPC, load balancer, or task-definition machinery,
which matters because judges must be able to open a working URL on demand and a
cold start on camera is a real presentation risk.

**Documented alternative: Lambda container image + Function URL**, which scales
to zero and is cheaper at idle, at the cost of cold-start latency on a large
image. Phase 4 will choose between them on measured cold-start and cost, and the
decision will be recorded here.

## 21. AWS service justification

| Service | Why it is required | Why not something else |
| --- | --- | --- |
| ECR | App Runner and Lambda both require a registry | Public registries cannot hold private build artefacts |
| App Runner | Container → HTTPS endpoint, minimal infrastructure | ECS/Fargate needs ALB, VPC, task defs for no added benefit here |
| S3 | Durable storage for uploads and evidence overlays | Local container disk is ephemeral |
| DynamoDB | Per-inspection trace records, serverless, cheap | RDS would require VPC and idle cost for a key-value workload |
| Bedrock | Managed model access for explanation | Self-hosting an LLM contradicts the simplicity goal |
| CloudWatch | Logs, metrics, alarms — the responsible-operation requirement | — |
| IAM | Least-privilege scoped roles | — |

**Explicitly rejected, to avoid cloud-service stuffing:** API Gateway (App Runner
already terminates HTTPS), Step Functions (the agent loop is in-process and
bounded), SageMaker endpoints (cost and complexity far exceed the need), SQS /
EventBridge / Kinesis (no asynchronous fan-out at this scale), VPC with NAT
(nothing requires private networking), Cognito (no accounts in scope).

## 22. Security and IAM considerations

- One scoped task role; no long-lived access keys anywhere in the repository.
- S3: bucket private, no public ACLs; presigned URLs, short expiry, for demo
  access to evidence images.
- Least privilege: `s3:GetObject`/`PutObject` on one prefix, `dynamodb` on one
  table, `bedrock:InvokeModel` on one model ARN.
- Upload validation: content-type and magic-byte checks, hard size cap, decode
  in a guarded path — image decoders are a real attack surface.
- Secrets via environment/Parameter Store, never committed. The `.gitignore`
  must be extended on this branch to cover `.env.*`, `*.pem`, `*.key` before any
  infrastructure work begins.
- Rate limiting on the public endpoint, and a spend alarm.

## 23. Observability

- Structured JSON logs, one event per agent step, correlated by inspection id.
- CloudWatch metrics: request count, end-to-end and per-stage latency, gate
  rejection rate, remediation rate, escalation rate, error rate, Bedrock spend.
- Alarms: error rate, p95 latency, unexpected cost.
- The trace record is the primary debugging artefact and is retrievable by id.

## 24. Data lifecycle

| Stage | Policy |
| --- | --- |
| Upload | Private S3 prefix, server-side encryption |
| Processing | In-memory; no persistent local copies |
| Evidence overlay | Stored under the inspection id |
| Trace record | DynamoDB, retrievable by inspection id |
| Retention | Lifecycle expiry on demo data; documented in the report |
| Deletion | Documented path to delete an inspection and its artefacts |

## 25. Dataset provenance and license boundaries

This is the most serious non-technical risk in the project.

| Dataset | License | Consequence |
| --- | --- | --- |
| Kaggle `sriramr/fruits-fresh-and-rotten-for-classification` | **Unknown** — no redistribution license found | Local evaluation only. **Must not** be redistributed in the repo, archive, or video. |
| Sultana external set | CC BY 4.0 | Usable with attribution. |
| FruitVision (Mendeley v2) | **CC BY-NC-ND 4.0** | NonCommercial **and** NoDerivatives. Do not redistribute images or derivatives. |

The research line already respects this: only identifiers, hashes, statistics,
predictions and aggregate plots are versioned, and no third-party images are in
the repository. **The competition line must hold the same boundary**, which is
harder because a public demo and a video both display images.

**Recommendation: self-capture the demo and video imagery.** Photographing
produce directly removes every redistribution question, and simultaneously
demonstrates the realistic capture conditions — poor light, glare, handheld
blur — that the system is built to handle. Licensed evaluation sets stay local
and are reported as numbers only.

## 26. Evaluation methodology

Full detail in [EVALUATION_PLAN.md](EVALUATION_PLAN.md). Principles:

- Metrics are defined **before** implementation and only if measurable.
- Competition measurements are reported separately from V2 research results.
- The controlled-degradation suite provides ground truth for perception and
  agent behaviour without any new labelling of produce condition.
- No result is claimed without an artefact in the repository supporting it.

## 27. Baselines

1. **Single-shot classifier, no agent** — the honest comparator; isolates the
   contribution of the agent loop.
2. **Perception gate without remediation** — isolates the contribution of acting
   versus merely detecting.
3. **V1 historical Keras CNN** — reported from existing records only, never re-run.
4. **Published V2 numbers** (Exp 014) — cited as the research context for
   out-of-domain degradation, not re-derived.

## 28. Failure-case evaluation

A curated failure set, self-captured and synthetically degraded, spanning:
blur, underexposure, overexposure, glare, partial occlusion, cluttered
background, multiple fruits, out-of-taxonomy produce, non-produce input, and
borderline fresh/rotten cases. Each case documents expected behaviour, observed
behaviour, and whether the system failed **safely** (escalated) or **unsafely**
(confidently wrong).

## 29. Performance evaluation

Per-stage and end-to-end latency (p50/p95), cold versus warm start, payload size
sensitivity, cost per inspection, and concurrency behaviour. Latency is measured
on the deployed AWS endpoint, not only locally.

## 30. Agentic task-success evaluation

The measurements that substantiate the Agentic Vision claim:

- **Decision attribution rate** — proportion of non-trivial decisions where a
  specific OpenCV metric crossing a specific threshold provably caused the
  branch. This is the direct evidence that perception changed behaviour.
- **Recovery rate** — degraded inputs where remediation restored a correct
  decision that the single-shot baseline got wrong.
- **Escalation precision / recall** — against the controlled-degradation ground
  truth of which inputs *should* have been escalated.
- **Unnecessary escalation rate** — the usability cost of caution.
- **Trace completeness** — proportion of inspections with a well-formed,
  replayable trace.
- **Calibration** — reliability diagram and expected calibration error.
- **Loop termination** — bound respected; no oscillation.

## 31. Responsible-use boundary

AgriVision assesses **visible produce condition only**. Permitted language:
"visible freshness condition", "surface deterioration", "visual quality
inspection", "surface anomaly", "human review recommended".

The system does **not** detect pathogens, toxins, microbiological contamination,
internal spoilage, or food safety, and makes no edibility claim. No such claim
will be made unless a future validated experiment specifically supports it.

This boundary appears in the UI, the API response, the report and the video —
not only in documentation. Known limitations, including the out-of-domain
degradation in Section 10, are stated plainly rather than omitted.

## 32. Demo scenario

A judge opens the endpoint and runs three cases:

1. **Clean capture** → accepted directly; evidence and trace shown.
2. **Poorly lit capture** → perception flags exposure, agent applies CLAHE,
   re-runs, and reaches a confident decision. The trace shows the metric that
   fired and the action it caused. *This is the Agentic Vision proof.*
3. **Blurred or conflicting capture** → agent exhausts bounded remediation and
   escalates for human review with a specific reason.

The contrast between cases 1 and 2 is the entire submission in one screen.

## 33. Video story (≤5 minutes)

| Segment | Duration | Content |
| --- | --- | --- |
| Problem | ~40s | Manual inspection; cost of confidently-wrong automation |
| Why perception first | ~40s | Out-of-domain degradation evidence from Section 10 |
| Architecture | ~40s | OpenCV → model → decision → action, on AWS |
| Live case 1 | ~30s | Clean capture accepted |
| Live case 2 | ~60s | Degraded capture; metric fires, action taken, trace shown |
| Live case 3 | ~40s | Escalation to human review |
| Evaluation & limits | ~40s | Headline numbers, failure cases, claim boundary |

## 34. Technical report outline

1. Problem and motivation
2. Prior work in this repository (V1, V2) and what it establishes
3. Why perception-first, with the out-of-domain evidence
4. OpenCV 5 pipeline design
5. Condition model selection and calibration
6. Agentic decision architecture and tool set
7. AWS deployment and reproducibility
8. Evaluation methodology and results
9. Failure analysis and limitations
10. Responsible use and claim boundary
11. Reproduction instructions
12. Provenance, licensing and attribution

## 35. Implementation phases

No dates are given: the official competition schedule has not been verified in
this branch. Dates will be added only from the official schedule.

### Phase 0 — Competition architecture and repository foundation *(this task)*
- **Objective:** isolated branch, blueprint, scaffold, dependency boundary.
- **Artifacts:** this document, traceability matrix, evaluation plan, submission
  checklist, `competition/` scaffold, `requirements-competition.txt`.
- **Tests:** none (no implementation).
- **Exit:** branch verified from `3928d43`; research lines untouched; docs resolve.
- **Supports:** documentation/presentation.

### Phase 1 — OpenCV 5 perception baseline *(capture-quality slice complete)*
- **Objective:** the perception layer and its evidence record.
- **Done:** capture-quality metrics (sharpness, illumination, contrast,
  clipping, validity); typed serialisable `PerceptionEvidence`; threshold policy
  separated from measurement; controlled-degradation generator; CLI; baseline
  sweep with plots and latency; 81 tests.
- **Remaining for later phases:** enhancement actions, segmentation, ROI and
  anomaly localisation, evidence overlay rendering.
- **Artifacts:** `competition/vision/`, `competition/evaluation/`,
  `tests/competition/`, [PHASE1_OPENCV_PERCEPTION.md](PHASE1_OPENCV_PERCEPTION.md).
- **Tests:** per-metric unit tests on synthetic fixtures of known degradation;
  monotonicity, determinism, safe-failure and path-hygiene tests.
- **Exit:** met for the capture-quality slice — every metric measured against
  controlled degradation and deterministic.
- **Supports:** technical execution, innovation.

### Phase 2 — Condition model integration
- **Objective:** select and wrap the model with calibrated uncertainty.
- **Tasks:** candidate comparison on retention and footprint; calibration;
  inference wrapper decoupled from research code.
- **Artifacts:** `competition/models/`, selection record with justification.
- **Tests:** deterministic inference, calibration regression, contract tests.
- **Exit:** model selected on recorded evidence; uncertainty calibrated.
- **Supports:** technical execution.

### Phase 3 — Agentic perception–decision–action workflow
- **Objective:** the real loop, with traces.
- **Tasks:** tool interfaces; deterministic policy engine; bounded remediation;
  escalation; trace schema; Bedrock explanation.
- **Artifacts:** `competition/agent/`, `competition/agent/tools/`, trace schema.
- **Tests:** policy unit tests per branch; end-to-end scenario tests asserting a
  named OpenCV metric caused a named action; termination and oscillation tests.
- **Exit:** decision attribution demonstrable on every scenario.
- **Supports:** **Agentic Vision Award**, innovation, technical execution.

### Phase 4 — AWS deployment and observability
- **Objective:** a reproducible public endpoint.
- **Tasks:** container build; App Runner vs Lambda decision on measured cold
  start; S3/DynamoDB/Bedrock wiring; IaC; IAM; logging, metrics, alarms.
- **Artifacts:** `infrastructure/aws/`, deployment guide, architecture diagram.
- **Tests:** smoke tests against the deployed endpoint; IAM least-privilege check.
- **Exit:** endpoint reachable; deployable from a clean account by the guide.
- **Supports:** AWS/reproducibility, real-world impact.

### Phase 5 — UI and judge demonstration
- **Objective:** a judge-operable interface showing evidence and trace.
- **Tasks:** upload, evidence overlay, trace timeline, escalation view, claim
  boundary surfaced in the UI.
- **Artifacts:** `competition/ui/`.
- **Tests:** interaction tests for the three demo scenarios.
- **Exit:** a judge can reproduce all three scenarios unaided.
- **Supports:** UX, presentation.

### Phase 6 — Evaluation and failure analysis
- **Objective:** execute the evaluation plan and report honestly.
- **Tasks:** degradation suite; baselines; agent metrics; latency; failure
  taxonomy; calibration.
- **Artifacts:** `competition/evaluation/`, results and figures.
- **Tests:** reproducible evaluation entry point, seeded.
- **Exit:** every claimed number traced to a committed artefact.
- **Supports:** technical execution, real-world impact.

### Phase 7 — Hardening, documentation and final submission
- **Objective:** ship it.
- **Tasks:** dependency pinning audit; security review; technical report;
  architecture diagram; video; Devpost materials; final reproduction rehearsal.
- **Artifacts:** report, diagram, video, completed checklist.
- **Exit:** every checklist item green with a linked artefact.
- **Supports:** all criteria.

## 36. Risks and mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| OpenCV 5 wheel unavailable on the container base image | High | Verify in Phase 1 container build; fall back to a base image that resolves it. Note `ximgproc` is already known unavailable. |
| Cold start degrades the live demo | High | App Runner as primary; measure in Phase 4; keep the image lean. |
| Dataset licensing blocks public demo imagery | High | Self-captured demo and video imagery (Section 25). |
| Agent loop reads as cosmetic to judges | High | Decision-attribution metric (Section 30) as explicit evidence. |
| Out-of-domain degradation embarrasses the demo | Medium | It is the *motivation*, stated openly; escalation is the designed response. |
| No existing tests or CI | Medium | Test infrastructure is an explicit Phase 1 deliverable, not an afterthought. |
| Bedrock model access or region availability | Medium | Verify access early in Phase 3; the deterministic policy must work without Bedrock. |
| AWS cost overrun | Low | Budget alarm, lifecycle expiry, scale-to-zero option. |
| Scope creep into research | Medium | Section 5 boundary; no retraining on this branch. |

## 37. Definition of done

- [ ] OpenCV 5 performs substantive analysis that measurably changes behaviour.
- [ ] A meaningful component runs on AWS, reachable by judges.
- [ ] Perception → decision → action loop proven by decision attribution.
- [ ] Technical report complete.
- [ ] Repository judge-accessible, dependencies pinned.
- [ ] Build, deployment and test instructions verified from a clean checkout.
- [ ] Architecture diagram published.
- [ ] Working endpoint or live demo available.
- [ ] Video ≤5 minutes, judge-accessible.
- [ ] Evaluation evidence committed, every number traceable to an artefact.
- [ ] Failure cases and limitations documented.
- [ ] Responsible-use boundary stated in UI, API, report and video.
- [ ] `legacy` and `master` unmodified by competition work.

---

## Related documents

- [REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md)
- [EVALUATION_PLAN.md](EVALUATION_PLAN.md)
- [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)
- [PHASE1_OPENCV_PERCEPTION.md](PHASE1_OPENCV_PERCEPTION.md) — Phase 1 technical note
