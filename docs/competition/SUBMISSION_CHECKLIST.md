# Submission Checklist

Nothing here may be ticked without a committed artifact or a working link. Every
item names where its evidence lives.

Base commit: `3928d43b3bdd3a754f98f1f411596050de29da17`
Branch: `competition/opencv-aws-2026`

---

## 1. OpenCV 5 evidence

- [ ] OpenCV 5 pinned in `requirements-competition.txt` and in the container image
- [ ] Installed version recorded from the running container, not just declared
- [ ] Perception layer performs substantive analysis (quality, segmentation, colour, texture, anomaly localisation)
- [ ] OpenCV operations are load-bearing — removing them changes system behaviour
- [ ] Per-metric tests against controlled degradations pass
- [ ] Perception determinism verified (identical input → identical evidence record)
- [ ] Ablation recorded: system behaviour with perception disabled

## 2. AWS evidence

- [x] A meaningful component runs on AWS — the full OpenCV 5 perception and
      Agentic Vision loop, as a container on App Runner
- [x] Live HTTPS endpoint with a managed certificate
- [x] Model artifact in S3, fetched at startup and SHA-256 verified, fail closed
- [x] Causal traces persisted to DynamoDB with a 14-day TTL
- [x] Structured JSON logs in CloudWatch carrying run_id, state, tool, action,
      reason_code and duration
- [x] IAM least privilege — the runtime role reads one S3 prefix and writes one
      table; no AdministratorAccess; no long-lived credential in the image
- [x] No public storage — S3 public access fully blocked, encrypted, versioned
- [x] Infrastructure as code — one CloudFormation template
- [x] Deployment smoke-tested against the live endpoint
- [x] Provisioning uses a scoped identity, not root — `user/agrivision-deployer`
      with one customer-managed least-privilege policy; verified by redeploying
      the service entirely under it, and by confirming `iam list-users` and
      `s3 ls` are both denied to it
- [x] No persistent root access keys exist on the account
      (`AccountAccessKeysPresent = 0`); root MFA enabled
- [ ] Endpoint authentication — currently public, acceptable for a judged demo
- [ ] Load or concurrency characterisation — none performed

> Deliberately not used: API Gateway, Lambda, Step Functions, SQS, EventBridge,
> SageMaker, VPC/NAT, Cognito, Bedrock. Each was considered; none had a measured
> requirement this service could point at.

## 3. Repository readiness

- [ ] Competition branch isolated; `legacy` and `master` unmodified
- [ ] Repository or archive accessible to judges
- [ ] README explains the branch structure and links the blueprint
- [ ] No secrets, credentials or `.env` files committed
- [ ] `.gitignore` covers `.env.*`, `*.pem`, `*.key` before any infrastructure work
- [ ] No third-party dataset images redistributed
- [ ] Attribution present for CC BY 4.0 material
- [ ] Repository size sane; no accidental large binaries

## 4. Pinned dependencies

- [ ] `requirements-competition.txt` fully pinned — no unresolved runtime entries
- [ ] Container base image pinned by digest
- [ ] Research requirements files unmodified by competition work
- [ ] Clean-environment install verified

## 5. Deployment guide

- [ ] Prerequisites listed
- [ ] Build, push, deploy steps verified end to end
- [ ] Configuration and environment variables documented
- [ ] Teardown instructions included
- [ ] Rehearsed from a clean checkout by following only the written steps

## 6. Architecture diagram

- [ ] Diagram reflects what is actually deployed
- [ ] Shows the perception → decision → action loop
- [ ] Shows AWS components and data flow
- [ ] Referenced from README and technical report

## 7. Demo endpoint

- [ ] Endpoint live and reachable
- [ ] Three demo scenarios reproducible by a judge unaided (Blueprint §32)
- [ ] Evidence overlay and trace visible in the UI
- [ ] Claim boundary visible in the UI
- [ ] Graceful behaviour on invalid, oversized and non-produce input
- [ ] Availability confirmed close to judging

## 8. Evaluation artifacts

- [ ] Controlled-degradation suite committed and regenerable from seed
- [ ] Baseline comparisons recorded
- [ ] Classification metrics with confidence intervals
- [ ] OpenCV detector metrics (ROC/AUC vs degradation ground truth)
- [ ] Latency measured on the deployed endpoint
- [ ] Agent task-success metrics recorded
- [ ] **Decision attribution rate** measured and reported
- [ ] Recovery, harm and escalation metrics recorded
- [ ] Calibration reported
- [ ] Every claimed number traceable to a committed artifact

## 9. Failure cases

- [ ] Failure taxonomy F1–F10 populated with real cases
- [ ] Unsafe failures (F8) reported explicitly and not omitted
- [ ] Curated failure set committed
- [ ] Limitations section written, including out-of-domain degradation
- [ ] Expected vs observed behaviour documented per case

## 10. Responsible AI

- [ ] Claim boundary stated in UI, API response, report and video
- [ ] No pathogen, toxin, contamination, internal-spoilage, safety or edibility claim anywhere
- [ ] Approved language used throughout ("visible freshness condition", "surface deterioration", "human review recommended")
- [ ] Human-review path documented and functioning
- [ ] Dataset licensing and redistribution boundary documented
- [ ] Data retention and deletion documented
- [ ] Known biases and domain limitations disclosed

## 11. Agentic Vision trace

- [x] Trace schema documented and versioned — `AgentTrace`, `phase3-agent-trace-1.0.0`
- [x] Every inspection produces a complete, replayable trace — completeness 12/12;
      timing excluded from the deterministic payload so runs compare byte-identical
- [x] Traces name the OpenCV metric, threshold and resulting action — e.g.
      `high_frequency_ratio 0.2635` against floor `0.3182` → `REQUEST_RECAPTURE`
- [x] Traces demonstrate tool calls caused by visual evidence — counterfactual:
      one base subject, identical policy, four distinct first actions
- [x] Traces demonstrate re-analysis caused by visual evidence — remediation forces
      re-segmentation and re-measurement; a second attempt is unreachable
- [x] Traces demonstrate human-approval requests caused by visual evidence — three
      human-action terminals, each carrying its triggering evidence and maturity
- [x] Control decisions are deterministic, not model-generated — no language model
      participates; `resolve_tool` refuses anything outside the closed vocabulary
- [x] System demonstrably works with Bedrock unavailable — no Bedrock exists, and
      the suite also runs with the ONNX artifact absent (636 passed, 94 skipped)
- [ ] Worked trace walkthrough included in the report — six curated traces exist in
      `results/phase3/traces/`; the report itself is Phase 7

> Evidence maturity is carried on every step. A reader can tell a `CALIBRATED`
> finding from an `ADVISORY` one without knowing which detectors were calibrated.

## 12. Technical report

- [ ] Follows the Blueprint §34 outline
- [ ] Prior V1/V2 work described and correctly attributed
- [ ] Design decisions justified with evidence
- [ ] Results reported honestly, including negative results
- [ ] Limitations and responsible use included
- [ ] Reproduction instructions included
- [ ] Citations and dataset attributions complete

## 13. Video (≤5 minutes)

- [ ] Under 5:00
- [ ] Judge-accessible link
- [ ] Follows the Blueprint §33 storyboard
- [ ] Shows the agent loop changing behaviour on real input
- [ ] Shows an escalation
- [ ] States the claim boundary
- [ ] Uses only imagery that is licensed or self-captured
- [ ] Audio and screen content legible

## 14. Final Devpost materials

- [ ] Project title and tagline
- [ ] Description covering problem, approach and impact
- [ ] Repository link
- [ ] Demo endpoint link
- [ ] Video link
- [ ] Technical report link
- [ ] Architecture diagram attached
- [ ] Award categories selected (Overall, Agentic Vision)
- [ ] Team and attributions complete
- [ ] Submitted before the deadline (schedule not yet verified — confirm from the official source)
