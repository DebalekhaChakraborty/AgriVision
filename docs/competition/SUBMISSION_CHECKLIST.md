# Submission checklist — operational

**Rule: an item is only ticked when a named artifact backs it.** "Looks done" is
not evidence. Every ticked row below names the file, command or measurement that
makes it true.

Status as of the end of Phase 7. Phase 7 is **uncommitted**.

---

## 1. Technical report and documentation

| Item | Status | Evidence |
| --- | --- | --- |
| Technical report | **DONE** | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md), 20 sections |
| Result labels used consistently | **DONE** | DEVELOPMENT / CONFIRMATORY / EXPLORATORY / RETROSPECTIVE ABLATION defined at the top and applied throughout |
| Canonical result table with denominators | **DONE** | TECHNICAL_REPORT §13 |
| Underexposure limitation given its own section | **DONE** | TECHNICAL_REPORT §15.1, not an appendix |
| Segmentation limitation stated | **DONE** | TECHNICAL_REPORT §15.2, §19 |
| Condition-model trade-off stated | **DONE** | TECHNICAL_REPORT §9 — 15.9 points worse than the best candidate, stated plainly |
| README oriented for a judge | **DONE** | README competition section; V1/V2 research record unchanged below it |
| Architecture diagram | **DONE** | [ARCHITECTURE.md](ARCHITECTURE.md) — deployed components only |
| Agent workflow diagram | **DONE** | [AGENT_WORKFLOW_DIAGRAM.md](AGENT_WORKFLOW_DIAGRAM.md) — bounds drawn on the diagram |
| Evaluation method | **DONE** | [PHASE6_FINAL_EVALUATION.md](PHASE6_FINAL_EVALUATION.md) |
| Build / deploy / test documentation | **DONE** | [PHASE4_AWS_DEPLOYMENT.md](PHASE4_AWS_DEPLOYMENT.md), README reproduction block |

## 2. Repository

| Item | Status | Evidence |
| --- | --- | --- |
| Judge-accessible repository | **DONE** | public remote, branch `competition/opencv-aws-2026` |
| Pinned dependencies | **DONE** | `requirements-competition.txt`, `requirements-serving.txt`; `results/phase7/dependency_freeze.json` |
| `pip check` clean in the container | **DONE** | "No broken requirements found" — dependency_freeze.json |
| No research dependency in the serving runtime | **DONE** | torch, torchvision, onnx, onnxruntime, matplotlib, pytest all absent — dependency_freeze.json |
| Dependency manifest / SBOM | **DONE** | `results/phase7/sbom.json`, 34 components, 0 licences guessed |
| No credentials in the repository | **DONE** | `results/phase7/security_audit.json` — 667 files, 31 commits, 0 findings |
| No restricted image bytes committed | **DONE** | corpus bytes gitignored; every committed image is a generated plot |
| Research history untouched | **DONE** | `git status` on `src/ v2/ model/ dataset/ notebooks/ results/ outputs/` → 0 changes |

## 3. Live demonstration

| Item | Status | Evidence |
| --- | --- | --- |
| Live endpoint reachable | **DONE** | <https://yp2ajauzkm.us-east-1.awsapprunner.com> |
| `/health` and `/ready` pass | **DONE** | `results/phase7/live_final_smoke.json` |
| Five demo scenarios behave as documented | **DONE** | live_final_smoke.json — 5/5 matched expected model invocation and blocking |
| Traces retrievable for every scenario | **DONE** | live_final_smoke.json |
| Policy fingerprints match the frozen system | **DONE** | live_final_smoke.json — all match |
| UI surface correct | **DONE** | responsible use, one-primary-fruit contract, no food-safety claim, counterfactual control, raw-trace disclosure — all verified in live_final_smoke.json |
| Human visual check in a real browser | **DONE** | `HUMAN_UI_SMOKE_CHECK = PASS`, desktop and narrow width; manual, not automated |
| Final demo revision recorded | **DONE** | `FINAL_DEMO_REVISION` in live_final_smoke.json — digest `sha256:b57eaf05…` |
| Service stable, no failed deployment | **DONE** | live_final_smoke.json — RUNNING, 0 failed operations |

## 4. Evidence

| Item | Status | Evidence |
| --- | --- | --- |
| OpenCV 5 substantive role | **DONE** | TECHNICAL_REPORT §5; metric selection recorded in PHASE2C_LICENSED_CALIBRATION |
| Agentic Vision evidence | **DONE** | 10/12 real bases change action; `results/phase6/counterfactual_matrix.json` |
| Agent trace evidence | **DONE** | 58/58 decision attribution; `results/phase6/track_c_results.json` |
| AWS evidence | **DONE** | `results/phase7/iac_check.json` — template matches live on 5/5 fields |
| Confirmatory evaluation | **DONE** | Phase 6, committed at `0124c6d` |
| Agent-disabled ablation | **DONE** | `results/phase7/baseline_ablation.json`, labelled RETROSPECTIVE / EXPLORATORY |
| Failure cases documented | **DONE** | `results/phase6/failure_taxonomy.json`; TECHNICAL_REPORT §15 |
| Limitations documented | **DONE** | TECHNICAL_REPORT §19; README limitations block |
| Responsible use stated | **DONE** | live page, README, TECHNICAL_REPORT §16 |
| Observability verified | **DONE** | 6/6 — `results/phase7/cloudwatch_audit.json` |

## 5. Reproducibility

| Item | Status | Evidence |
| --- | --- | --- |
| Clean-clone rehearsal, all 13 steps | **DONE** | `results/phase7/clean_clone_reproduction.json` |
| Deployed policy fingerprint reproduced | **DONE** | `78b1e2151773787a` reproduced in the clean clone |
| Artifact-independent test mode | **DONE** | 770 passed, 141 skipped, 0 failed without the ONNX file |
| Full test suite | **DONE** | 911 passing locally; 909 passed / 2 skipped in the clean clone |
| Model checksum verification | **DONE** | fail-closed; verified in the clean-clone container |

## 6. Image licensing

| Item | Status | Evidence |
| --- | --- | --- |
| Per-image licence verified for the evaluation corpus | **DONE** | `results/phase6/phase6_licence_manifest.json` — 38 entries, all commercial + derivatives, no Unknown/NC/ND |
| Public-facing attribution list | **DONE** | [IMAGE_ATTRIBUTIONS.md](IMAGE_ATTRIBUTIONS.md) — currently nothing to attribute; no third-party photograph is shown publicly |
| Demo asset policy | **DONE** | IMAGE_ATTRIBUTIONS.md and VIDEO_RECORDING_CHECKLIST.md |

## 6b. Official competition requirements

Checked against the stated rules. **Video maximum: 5 minutes.**

| Requirement | Status | Evidence |
| --- | --- | --- |
| Technical report | **DONE** | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) |
| Judge-accessible code repository | **DONE** | public remote, branch `competition/opencv-aws-2026` |
| Pinned dependencies | **DONE** | both requirements files; `dependency_freeze.json` |
| Build instructions | **DONE** | README reproduction block; `Dockerfile` |
| Deployment instructions | **DONE** | [PHASE4_AWS_DEPLOYMENT.md](PHASE4_AWS_DEPLOYMENT.md) |
| Test instructions | **DONE** | README; artifact-absent mode documented |
| Architecture diagram | **DONE** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Working endpoint | **DONE** | <https://yp2ajauzkm.us-east-1.awsapprunner.com> |
| Video, public or unlisted, judge-accessible | **NOT DONE** | a person records this |
| Video ≤ 5 minutes | **NOT DONE** | script targets 4:00, leaving a minute of margin |
| Application working in video | **NOT DONE** | script §0:50 and §1:20 are live runs |
| Architecture shown in video | **NOT DONE** | script §2:50 |
| Principal results shown in video | **NOT DONE** | script §3:15 |
| Evaluation evidence | **DONE** | `results/phase6/`, PHASE6_FINAL_EVALUATION.md |
| Failure cases and limitations | **DONE** | TECHNICAL_REPORT §15, §19 |
| Responsible use | **DONE** | live page, README, TECHNICAL_REPORT §16 |

### Agentic Vision — additional evidence

| Requirement | Status | Evidence |
| --- | --- | --- |
| Workflow diagram | **DONE** | [AGENT_WORKFLOW_DIAGRAM.md](AGENT_WORKFLOW_DIAGRAM.md) |
| Trace where OpenCV output changes a later decision | **DONE** | 10/12 real bases change action; `counterfactual_matrix.json`; TECHNICAL_REPORT §7 |
| Task-success evaluation | **DONE** | 29/48 preregistered; `track_c_results.json` |
| Failure handling | **DONE** | fail-safe 12/12; `failure_taxonomy.json` |
| Observability | **DONE** | 6/6 verified; `cloudwatch_audit.json` |
| Human control | **DONE** | three human-action terminals; requests are guidance, never simulated as performed |

## 7. Video — OUTSTANDING

| Item | Status | Evidence |
| --- | --- | --- |
| Demo script | **DONE** | [JUDGE_DEMO_SCRIPT.md](JUDGE_DEMO_SCRIPT.md), 4:00 target |
| Recording checklist | **DONE** | [VIDEO_RECORDING_CHECKLIST.md](VIDEO_RECORDING_CHECKLIST.md) |
| Competition maximum length confirmed | **NOT DONE** | needs checking against current rules |
| Video recorded | **NOT DONE** | a person records this; it is not automated |
| Video within the length limit | **NOT DONE** | |
| Judge-accessible video link | **NOT DONE** | record the URL here once uploaded: `______` |
| Demo image attributed if third-party | **NOT DONE** | add a row to IMAGE_ATTRIBUTIONS.md **before** recording |

## 7b. Release record

Filled in when the tag is created.

| Field | Value |
| --- | --- |
| Tag | `opencv-aws-2026-submission` |
| Commit SHA | _(recorded at tag time)_ |
| Branch | `competition/opencv-aws-2026` |
| Date | _(recorded at tag time)_ |
| Live service | <https://yp2ajauzkm.us-east-1.awsapprunner.com> |
| Live revision digest | `sha256:b57eaf0506a243798ef21423048e1c571606d78243caa13bbf7d11b79d6ce7e7` |
| Container digest (ECR `phase4`) | same as above |

## 8. Submission mechanics — OUTSTANDING

| Item | Status | Evidence |
| --- | --- | --- |
| Phase 7 committed | **NOT DONE** | held for final review |
| Release tag created | **NOT DONE** | candidate name and procedure in the Phase 7 report; do not tag without authorisation |
| Devpost / submission form completed | **NOT DONE** | |
| Final reproduction rehearsal from the tagged revision | **NOT DONE** | the Phase 7 rehearsal ran against `0124c6d`, not a tag |

---

Remaining human-only actions: [HUMAN_TODO.md](HUMAN_TODO.md).

## Known gaps carried into submission

Stated here so they are decisions rather than oversights:

1. **Visible-condition accuracy on real imagery is unmeasured.** No independent
   label exists. Fruit type is a proxy for domain fit, not the product claim.
2. **The quality gate's benefit is unproven on the task it exists for.** On
   fruit type it cost coverage and prevented no error (TECHNICAL_REPORT §14).
3. **The model artifact is not obtainable by a third party** from the public
   repository alone. The artifact-independent suite runs for anyone; the
   container does not.
4. **No phone-camera validation** has been collected.
5. **The endpoint is unauthenticated**, which is acceptable for a judged demo.
