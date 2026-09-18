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
| A1 | Use OpenCV 5 for substantive image/video analysis | Perception layer plus `cv2.dnn` inference (Blueprint §12) | Source showing core OpenCV operations doing load-bearing work; per-metric tests; ablation | `competition/vision/`, `competition/models/`, `tests/competition/` | 1, 1b, 2 | IN PROGRESS — capture quality, remediation and ONNX inference through cv2.dnn done, 259 tests; segmentation and localisation outstanding |
| A2 | Run a meaningful component on AWS | Container on App Runner (or Lambda container) running perception + agent loop; S3, DynamoDB, Bedrock, CloudWatch | Live endpoint; deployment logs; CloudWatch metrics | `infrastructure/aws/`, deployment guide | 4 | NOT STARTED |
| A3 | Technical report | Written to Blueprint §34 outline | Complete report document | `docs/competition/TECHNICAL_REPORT.md` | 7 | NOT STARTED |
| A4 | Judge-accessible code repository/archive | Public repository at the competition branch, plus tagged archive | Working clone URL; clean-checkout reproduction rehearsal | Repository + release archive | 7 | NOT STARTED |
| A5 | Pinned dependencies | `requirements-competition.txt` with exact versions; container lockfile | No unpinned runtime dependency; resolvable from a clean environment | `requirements-competition.txt` | 1→7 | IN PROGRESS — serving set pinned and separated from build-time; fastapi/uvicorn/pydantic/boto3 still unresolved |
| A6 | Build / deployment / test instructions | Step-by-step guide verified from a clean checkout and clean AWS account | Successful reproduction rehearsal record | `docs/competition/DEPLOYMENT.md` | 4→7 | NOT STARTED |
| A7 | Architecture diagram | Rendered diagram of the deployed system | Published image referenced by report and README | `docs/competition/architecture.*` | 4 | NOT STARTED |
| A8 | Working endpoint or live demo | App Runner HTTPS endpoint with the demo UI | Reachable URL; the three demo scenarios reproducible | Deployed service, `competition/ui/` | 4→5 | NOT STARTED |
| A9 | Video ≤5 minutes, judge-accessible | Recorded to Blueprint §33 storyboard | Hosted video under 5:00 | Video link in submission | 7 | NOT STARTED |
| A10 | Evaluation evidence | Execute EVALUATION_PLAN.md | Committed metrics, figures, and the code that produced them | `competition/evaluation/`, results | 6 | IN PROGRESS — Phase 1 sweep, Phase 1b experiments, and Phase 2 inventory/benchmarks/parity/condition evaluation produced |
| A11 | Failure cases and limitations | Failure taxonomy and curated failure set (Blueprint §28) | Documented cases with expected vs observed behaviour | `docs/competition/FAILURE_ANALYSIS.md` | 6 | NOT STARTED |
| A12 | Responsible use discussion | Claim boundary in UI, API response, report and video (Blueprint §31) | Boundary text present in all four surfaces | Report, UI, API schema | 5→7 | NOT STARTED |

## B. Agentic Vision Award requirements

| # | Requirement | Proposed implementation | Evidence required | Planned artifact | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | OpenCV 5 visual evidence influences a later decision | Deterministic policy engine consuming `PerceptionEvidence` (Blueprint §15, §16) | Trace records naming the metric, the threshold crossed, and the branch taken | Trace schema, `competition/agent/` | 1b, 3 | IN PROGRESS — capture-quality evidence selects the action; not yet extended to inspection decisions |
| B2 | Evidence influences a tool call | Perception metrics select which tool runs next (enhance, re-segment, reclassify ROI) | Trace showing tool invocation caused by a metric | Trace records | 1b, 3 | IN PROGRESS — gamma vs CLAHE vs recapture selected from evidence; tool set still small |
| B3 | Evidence influences a plan or re-analysis step | Bounded remediation loop re-enters perception with new parameters | Trace showing a second pass with changed parameters and the reason | Trace records | 1b, 3 | IN PROGRESS — single bounded re-assessment implemented; multi-step planning outstanding |
| B4 | Evidence influences a human-approval request | Escalation policy driven by capture quality, conflict, and uncertainty (Blueprint §17) | Escalation records with the triggering evidence attached | DynamoDB queue records | 3 | NOT STARTED |
| B5 | Not merely a chatbot explaining a fixed prediction | Control decisions are deterministic, not model-generated; Bedrock narrates only | Policy source; tests proving behaviour without Bedrock available | `competition/agent/`, policy tests | 1b, 3 | IN PROGRESS — policy is deterministic with no model dependency; no LLM present at all yet |
| B6 | The trace proves OpenCV changed what happened next | Decision-attribution metric (Blueprint §30) | Measured attribution rate over the evaluation set | Evaluation results | 6 | NOT STARTED |

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
