# Requirements Traceability Matrix

Every explicit final-submission requirement and every relevant Agentic Vision
requirement, mapped to its planned implementation, the evidence that will prove
it, and the artifact that will carry that evidence.

Status values: `NOT STARTED` · `IN PROGRESS` · `BLOCKED` · `COMPLETE`

No row may be marked `COMPLETE` without a committed artifact at the stated path.
Rows advanced by Phase 1 name what is done and what remains.

Base commit: `3928d43b3bdd3a754f98f1f411596050de29da17`

---

## A. Mandatory final-submission requirements

| # | Requirement | Proposed implementation | Evidence required | Planned artifact | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Use OpenCV 5 for substantive image/video analysis | Perception layer plus `cv2.dnn` inference (Blueprint §12) | Source showing core OpenCV operations doing load-bearing work; per-metric tests; ablation | `competition/vision/`, `competition/models/`, `tests/competition/` | 1, 1b, 2, 2b, 2c-B, 2d, 3 | IN PROGRESS — capture quality, remediation, `cv2.dnn` inference, foreground isolation, ROI policy calibrated on licensed real imagery, local artefact evidence, and a bounded agent loop over all of it; 730 tests. Anomaly localisation outstanding |
| A2 | Run a meaningful component on AWS | Container on App Runner running perception + agent loop; S3, DynamoDB, CloudWatch | Live endpoint; deployment logs; CloudWatch metrics | `infrastructure/aws/`, `competition/service/` | 4 | **SUBSTANTIALLY COMPLETE (Phase 4)** — the full OpenCV 5 perception and Agentic Vision loop runs on App Runner at `https://yp2ajauzkm.us-east-1.awsapprunner.com`, with the model fetched and SHA-256 verified from S3, traces in DynamoDB and structured logs in CloudWatch. Bedrock not used and not required. |
| A3 | Technical report | Written to Blueprint §34 outline | Complete report document | `docs/competition/TECHNICAL_REPORT.md` | 7 | NOT STARTED |
| A4 | Judge-accessible code repository/archive | Public repository at the competition branch, plus tagged archive | Working clone URL; clean-checkout reproduction rehearsal | Repository + release archive | 7 | NOT STARTED |
| A5 | Pinned dependencies | `requirements-competition.txt` with exact versions; container base pinned by digest | No unpinned runtime dependency; resolvable from a clean environment | `requirements-competition.txt`, `requirements-serving.txt`, `Dockerfile` | 1→7 | **COMPLETE (Phase 4)** — fastapi 0.141.1, uvicorn 0.53.0, pydantic 2.13.5, boto3 1.43.98, python-multipart 0.0.32 resolved and pinned; `pip check` clean; base image pinned by digest. |
| A6 | Build / deployment / test instructions | Step-by-step guide verified from a clean checkout and clean AWS account | Successful reproduction rehearsal record | `docs/competition/DEPLOYMENT.md` | 4→7 | NOT STARTED |
| A7 | Architecture diagram | Rendered diagram of the deployed system | Published image referenced by report and README | `docs/competition/DEPLOYED_ARCHITECTURE.md` | 4 | **COMPLETE (Phase 4)** — source-controlled Mermaid flowchart and sequence diagram of the deployed topology; only services actually used appear. |
| A8 | Working endpoint or live demo | App Runner HTTPS endpoint with the demo UI | Reachable URL; the three demo scenarios reproducible | Deployed service, `competition/ui/` | 4→5 | NOT STARTED |
| A9 | Video ≤5 minutes, judge-accessible | Recorded to Blueprint §33 storyboard | Hosted video under 5:00 | Video link in submission | 7 | NOT STARTED |
| A10 | Evaluation evidence | Execute EVALUATION_PLAN.md | Committed metrics, figures, and the code that produced them | `competition/evaluation/`, results | 6 | IN PROGRESS — Phase 1, 1b, 2, 2b, 2c-B, 2d and 3 evaluations produced; agent behaviour measured on a deterministic scenario suite, real-domain evaluation outstanding |
| A11 | Failure cases and limitations | Failure taxonomy and curated failure set (Blueprint §28) | Documented cases with expected vs observed behaviour | `docs/competition/FAILURE_ANALYSIS.md` | 6 | NOT STARTED |
| A12 | Responsible use discussion | Claim boundary in UI, API response, report and video (Blueprint §31) | Boundary text present in all four surfaces | Report, UI, API schema | 5→7 | NOT STARTED |

## B. Agentic Vision Award requirements

| # | Requirement | Proposed implementation | Evidence required | Planned artifact | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | OpenCV 5 visual evidence influences a later decision | Deterministic policy engine consuming `PerceptionEvidence` (Blueprint §15, §16) | Trace records naming the metric, the threshold crossed, and the branch taken | Trace schema, `competition/agent/` | 1b, 3 | **SUBSTANTIALLY COMPLETE (Phase 3)** — counterfactual experiment: one base subject, identical policy fingerprints, four distinct first actions. `results/phase3/counterfactual_actions.json`. Real-domain rates still unmeasured. |
| B2 | Evidence influences a tool call | Perception metrics select which tool runs next (enhance, re-segment, reclassify ROI) | Trace showing tool invocation caused by a metric | Trace records | 1b, 3 | **SUBSTANTIALLY COMPLETE (Phase 3)** — closed 15-tool registry; a blur finding selects `request_recapture`, underexposure selects `apply_gamma_correction`, contrast loss selects `apply_clahe`. Unnecessary tool-call rate 0/12. |
| B3 | Evidence influences a plan or re-analysis step | Bounded remediation loop re-enters perception with new parameters | Trace showing a second pass with changed parameters and the reason | Trace records | 1b, 3 | **SUBSTANTIALLY COMPLETE (Phase 3)** — remediation forces re-segmentation and re-measurement; the state machine has no edge permitting a second attempt. Multi-step planning beyond one excursion remains out of scope. |
| B4 | Evidence influences a human-approval request | Escalation policy driven by capture quality, conflict, and uncertainty (Blueprint §17) | Escalation records with the triggering evidence attached | DynamoDB queue records | 3 | **IN PROGRESS (Phase 3)** — three human-action terminals (`REQUEST_RECAPTURE`, `REQUEST_REPOSITION_LIGHT`, `REQUEST_HUMAN_REVIEW`) each carry their triggering evidence and maturity. Runs stop and wait; no durable queue exists yet (Phase 4). |
| B5 | Not merely a chatbot explaining a fixed prediction | Control decisions are deterministic, not model-generated; Bedrock narrates only | Policy source; tests proving behaviour without Bedrock available | `competition/agent/`, policy tests | 1b, 3 | **COMPLETE (Phase 3)** — no language model participates at any point. Actions are enum members; `resolve_tool` refuses any string outside the closed vocabulary. The renderer is a formatter. |
| B6 | The trace proves OpenCV changed what happened next | Decision-attribution metric (Blueprint §30) | Measured attribution rate over the evaluation set | Evaluation results | 6 | **IN PROGRESS (Phase 3)** — attribution 33/33 and trace completeness 12/12 on the deterministic scenario suite. A real-imagery attribution rate is Phase 6. |

## C. Judging-criteria coverage

| Criterion | Weight | Primary evidence | Phase |
| --- | --- | --- | --- |
| Technical execution | 30% | A1, A5, A10, B1–B6; test suite | 1,2,3,6 |
| Innovation | 20% | Perception-driven control; decision attribution | 1,3 |
| Real-world impact | 20% | Out-of-domain motivation (Blueprint §10); escalation; A11 | 3,6 |
| UX | 10% | A8, demo scenarios, evidence and trace surfaced | 5 |
| Documentation / presentation | 10% | A3, A7, A9, this matrix | 0,4,7 |
| AWS / reproducibility / responsible operation | 10% | A2, A5, A6, A12; IAM, CloudWatch | 4,7 |

## D. Repository foundation (Phase 0)

| # | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| D1 | Competition work isolated from research | Branch `competition/opencv-aws-2026` from `3928d43` | COMPLETE |
| D2 | `legacy` unmodified | `legacy` remains `9769e3c` | COMPLETE |
| D3 | `master` unmodified by competition work | `master` remains `3928d43` | COMPLETE |
| D4 | Blueprint established | `OPENCV_AWS_2026_BLUEPRINT.md` | COMPLETE |
| D5 | Traceability established | this document | COMPLETE |
| D6 | Evaluation defined before implementation | `EVALUATION_PLAN.md` | COMPLETE |
| D7 | Submission checklist established | `SUBMISSION_CHECKLIST.md` | COMPLETE |
| D8 | Scaffold created without moving research code | `competition/`, `infrastructure/aws/`, `tests/competition/` | COMPLETE |
| D9 | Dependency boundary declared | `requirements-competition.txt` | COMPLETE |
| D10 | Secret-bearing paths ignored before AWS work | `.gitignore` extension for `.env.*`, `*.pem`, `*.key`, cloud credential paths | COMPLETE |
| D11 | Competition environment isolated from the research venv | `.venv-competition`, gitignored; `.venv-v2` unmodified | COMPLETE |
| D12 | Model weights excluded from version control | `*.onnx` gitignored; only the artifact manifest is committed | COMPLETE |
| D13 | Competition demo imagery plan exists | `DEMO_DATA_PLAN.md`; no restricted imagery committed | COMPLETE |
| D14 | Deployment-domain capture protocol locked before collection | `PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md`; split fingerprint `2feb9360e1c2bdf5` generated pre-capture | COMPLETE |
| D15 | Calibration methodology fixed before data exists | `calibrate_quality_policy.py --show-method`; contains no threshold values, refuses to run without captures | COMPLETE |
| D16 | Self-captured imagery collected | **186** captures across 24 items (144 + 36 + 6) | **NOT STARTED — blocked on physical capture** |


## E. Dataset licensing and provenance (Phase 2c-B)

| Requirement | Where it is satisfied | Evidence |
| --- | --- | --- |
| Every third-party image has a verified licence | `competition/data/licence_validation.py` — default-deny gate run on metadata before download | `manifests/licensed_corpus.json`, `manifests/licence_rejections.json` |
| No Unknown, NonCommercial or NoDerivatives source is used | Allow-list of 10 CC/PD identifiers; NC and ND rejected by name | `test_licensed_corpus.py::test_non_commercial_licences_are_rejected`, `::test_no_derivatives_licences_are_rejected` |
| Attribution is preserved per image, not per dataset | `LicensedImageRecord.attribution_text`, required when the licence demands it | `::test_attribution_is_per_image_not_per_source` |
| Share-alike obligations are tracked | `share_alike_required` per record; 62 of 92 images carry it | licence table in [PHASE2C_LICENSED_CALIBRATION.md](PHASE2C_LICENSED_CALIBRATION.md) §4 |
| Evaluation is free of near-duplicate leakage | Split unit is the source group, never the image | `::test_a_built_split_never_lets_a_group_cross`, fingerprint `77a995ceff0d33f7` |
| Held-out data is opened only after the policy is frozen | `CorpusView` refuses a held-out view without a policy fingerprint | `::test_a_held_out_view_requires_a_policy_fingerprint` |
| A post-hoc threshold change cannot be reported as confirmatory | Append-only run ledger; a new threshold set after a held-out run is EXPLORATORY | `::test_changing_thresholds_after_a_held_out_run_makes_the_next_run_exploratory` |
| Generated imagery cannot be presented as real | Schema forbids a real licence or a real track on a `SYNTHETIC_GENERATED` record | `::test_a_generated_image_cannot_claim_a_real_licence` |
| No third-party image bytes are committed | `raw/`, `cache/`, `derived/`, `synthetic/` gitignored | `::test_image_directories_are_gitignored` |
| No machine-specific paths reach committed artifacts | `_reject_paths` on every record; manifests scanned | `::test_no_committed_manifest_contains_a_machine_path` |
| Claims stay inside the evidence | Locked policy status names which thresholds are provisional; claim boundary recorded in every results file | §11 and §16 of the phase note |


## F. Capture-artefact evidence (Phase 2d)

| Requirement | Where it is satisfied | Evidence |
| --- | --- | --- |
| Glare is measured locally, not by whole-image clipping | `competition/vision/highlights.py` — multi-scale local excess plus ROI-relative desaturation | `test_phase2d_artifacts.py::test_a_bright_saturated_fruit_is_not_called_glare` |
| Bright fruit is not called glare | Candidate pixels must be locally bright **and** desaturated relative to the subject's own chroma | `::test_a_uniformly_brightened_subject_is_not_called_glare` |
| Local evidence never includes background | Candidates are a subset of the eroded foreground | `::test_glare_candidates_never_include_background` |
| No unsupported semantic occlusion claim | Evidence is named `SUBJECT_VISIBILITY_INSUFFICIENT` and describes the region | `::test_no_semantic_occlusion_claim_appears_in_the_evidence` |
| Untrusted segmentation stops local evidence | `decide_capture_artifacts` escalates when the mask fails its guards | `::test_invalid_segmentation_prevents_trusted_artifact_evidence` |
| Actions come from a closed enum | `RemediationAction` / `ReasonCode`; no free-form strings | `::test_every_decision_uses_the_closed_action_enum` |
| Glare never triggers an enhancement that fakes recovery | Policy routes glare to `REQUEST_REPOSITION_LIGHT` only | `::test_glare_never_routes_to_an_enhancement` |
| Detectors below the preregistered bar cannot gate | `glare_blocks` / `visibility_blocks` default False | `::test_a_sub_floor_detector_records_its_finding_without_blocking` |
| Validation data is independent of calibration | No shared source group, no previously screened file, disjoint categories | `build_validation_pool.py`, fingerprint `e9e7ace82c40783e` |
| Parameters frozen before validation is opened | Fingerprints recorded in `results/phase2d/locked_artifact_policy.json` | `phase2d_artifacts.py::load_frozen_policies` verifies them |
| Latency is not described as AWS | `"environment"` field says local CPU | `results/phase2d/latency.json` |
