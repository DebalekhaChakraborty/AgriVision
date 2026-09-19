# Evaluation Plan

Defined **before** implementation, per the Phase 0 requirement. Every metric
below is measurable with resources the project already has or can create
cheaply. Metrics that sound impressive but cannot be honestly measured are
listed in the final section as explicitly excluded, with the reason.

Competition measurements are reported **separately** from V2 research results and
never merged into them. No number is claimed without a committed artifact
producing it.

---

## 1. Measurement substrate: the controlled-degradation suite

The keystone of this plan. Rather than labelling new produce data, known
degradations are applied to images whose condition label is already established:

| Degradation | Method | Swept parameter | Gives ground truth for |
| --- | --- | --- | --- |
| Defocus blur | Gaussian / disc kernel | kernel radius | blur detection, recapture policy |
| Motion blur | directional kernel | length, angle | blur detection |
| Underexposure | gamma / gain reduction | stop reduction | exposure detection, CLAHE benefit |
| Overexposure | gain increase with clipping | clipped fraction | exposure detection |
| Glare | synthetic specular blobs | area, intensity | glare detection, re-segmentation |
| Low contrast | histogram compression | compression factor | enhancement benefit |
| Occlusion | masked rectangles | occluded fraction | segmentation robustness, escalation |
| Noise | Gaussian / salt-and-pepper | sigma, density | denoise benefit |
| Downscale | resample | scale factor | resolution floor |

Because the degradation parameter is known, each image carries a derived label
for *should the system have flagged, remediated, or escalated this?* — which is
exactly the ground truth agent evaluation needs, and it requires no new
condition labelling.

**Caveat, stated plainly:** synthetic degradation is a proxy for real capture
failure. Self-captured real degraded images (Blueprint §25) form a smaller
companion set to check that conclusions transfer, and any divergence is reported.

## 2. Baseline metrics

| Baseline | Purpose | Source |
| --- | --- | --- |
| Single-shot classifier, agent disabled | Isolates the agent's contribution | **NOT measured.** Phase 6 measured the deployed agent, not an ablation of it. Outstanding. |
| Perception gate, detection only, no remediation | Isolates acting vs merely detecting | **NOT measured.** Outstanding. |
| V1 Keras CNN | Historical reference | **Cited from existing records; never re-run** |
| V2 Experiment 014 systems | Out-of-domain context | **Cited from `v2/results/experiment_014_multi_domain/`; never re-derived** |

## 2z. Status after Phase 6

The confirmatory evaluation is done and its numbers are in
[PHASE6_FINAL_EVALUATION.md](PHASE6_FINAL_EVALUATION.md). Measured on 38 fresh
licence-verified photographs and 48 controlled scenarios, on the **deployed AWS
service**, under a system frozen beforehand (`97b059be36bc3fbe`):

- foreground-valid 29/38, complete-inspection 28/38, fruit-type 27/28
- action selection 30/48 primary, 27/36 on bases whose reference segmented
- decision attribution 58/58, unsafe inference 0/86, fail-safe 12/12
- 10 of 12 bases change their action when only the visual condition changes

Two baselines listed in §2 remain unmeasured: the agent-disabled single-shot
classifier and the detection-only gate. Phase 6 measured the system as deployed
rather than ablations of it, so the agent's contribution is demonstrated
causally by the counterfactual matrix but is not yet quantified against a
no-agent baseline. That is the clearest remaining gap in this plan.

Still unmeasured for the reason given in §14 rather than by omission: visible
condition accuracy on real imagery, because no independent label exists.

## 2a. Status after Phase 2

Measured so far: OpenCV detector response across controlled degradations
(Phase 1), remediation behaviour and blur-verdict agreement (Phase 1b), and
condition accuracy, runtime parity and latency for the selected model
(Phase 2). See the phase notes for numbers.

Two plan assumptions have been corrected by measurement:

* **Calibration cannot use the research photographs as a proxy for deployment.**
  The capture gate rejects 96.7% of them because whole-image clipping statistics
  are dominated by bright backgrounds. Threshold calibration needs a development
  set drawn from the deployment domain — self-captured imagery
  ([DEMO_DATA_PLAN.md](DEMO_DATA_PLAN.md)).
* **Calibration is no longer blocked.** Phase 2c-B calibrated the ROI policy on
  92 licence-verified real photographs with controlled degradations
  ([PHASE2C_LICENSED_CALIBRATION.md](PHASE2C_LICENSED_CALIBRATION.md)). Splits
  are assigned per **source group**, the held-out set is fixed by fingerprint
  `77a995ceff0d33f7`, and thresholds are reported as a false-accept /
  false-block trade-off rather than a single accuracy figure. Self-capture
  remains available as an optional camera-domain validation and blocks nothing.
* **Capture-quality thresholds are scope-specific.** Phase 2b established that
  whole-image and foreground-restricted measurement are not interchangeable: ROI
  Laplacian variance was about 5% of the whole-image value because the latter was
  dominated by the subject outline. Any threshold set must state the scope it was
  calibrated for. Phase 2c-B confirmed the mechanism on an independent corpus:
  restricted to the subject, highlight clipping is **22× rarer** than
  whole-image, which is why the Phase 1 clipping limits fire on the backdrop.
* **Variance of the Laplacian cannot carry a global blur threshold, and this is
  now measured rather than suspected.** It separates blurred from sharp
  perfectly (AUC 1.000) yet its median moves **88-fold** across the exposure
  ladder on the same photographs. Blur gating uses `high_frequency_ratio`
  instead — AUC 0.999 with 1.13×/1.10×/1.03× movement across exposure and
  subject scale. Laplacian variance remains valid for *within-image* comparisons
  such as the Phase 1 degradation sweeps.
* **Some thresholds have no defensible value and are labelled as such.** Four of
  seven ROI thresholds met the preregistered rule; three retain provisional
  Phase 1 values, and the locked policy's status says so. A partially calibrated
  policy reported as calibrated would be the substitution Phase 2b warns about.
* **FruitVision is excluded from competition evaluation.** Its CC BY-NC-ND 4.0
  terms make competition-context use ambiguous. External-domain evaluation will
  use the CC BY 4.0 Sultana set and self-captured imagery.

## 3. Classification metrics

Over the six-class taxonomy, on the source test partition and each external
domain, following the existing research protocol so numbers are comparable:

- Accuracy; weighted F1; macro F1
- Per-class precision / recall / F1 (rotten-class recall matters most — a missed
  rotten item is the costly error)
- Confusion matrix
- Bootstrap confidence intervals, matching the research convention
- Retention: external accuracy ÷ source accuracy

## 4. OpenCV-specific metrics

Each perception metric is evaluated as a detector against the degradation sweep:

| Metric | Evaluation | Target property |
| --- | --- | --- |
| Sharpness (Laplacian variance) | ROC/AUC vs blur ground truth | Monotonic in blur radius |
| Exposure scores | ROC/AUC vs exposure ground truth | Symmetric under/over sensitivity |
| Glare fraction | ROC/AUC vs synthetic glare | Robust to bright-but-matte surfaces |
| Segmentation confidence | Correlation with IoU | Predicts mask quality without ground truth |
| Anomaly localisation | Precision/recall of regions | Regions overlap visible deterioration |
| Determinism | Repeat runs, same input | Byte-identical evidence record |
| Threshold sensitivity | Sweep each threshold | Documented operating points, not tuned post hoc |

## 5. Localisation and segmentation metrics

Valid only where ground truth exists, so scope is deliberately limited:

Phase 2b status: IoU and Dice are measured on **synthetic** fixtures with
constructed masks (mean IoU 0.905). No real-image segmentation IoU is reported,
because no trustworthy manual masks exist; real-image behaviour is reported as
validity rate and geometry distributions instead.

- **Segmentation IoU / Dice** against a small manually annotated mask set
  (target ~100 images, annotation effort explicitly budgeted in Phase 6). If
  annotation is not completed, IoU is **not** reported and segmentation
  confidence is reported as a correlation only.
- **Anomaly region precision/recall** against annotated deterioration regions on
  the same subset.
- Reported with the subset size and its selection method, never extrapolated.

## 6. Image-quality metrics

- Distribution of each quality score across clean vs degraded inputs.
- Gate rejection rate at the chosen operating point.
- **Enhancement benefit**: change in classification correctness and confidence
  after CLAHE/gamma, measured on degraded inputs where the clean-image label is
  known. This is the measurement justifying enhancement as an action rather than
  decoration.

## 7. Latency and processing time

Measured locally and on the deployed endpoint:

- Per-stage: decode, perception, model inference, decision, trace write
- End-to-end p50 / p95 / p99
- Cold start vs warm (the App Runner vs Lambda decision in Blueprint §20)
- Latency as a function of remediation depth
- Throughput under modest concurrency
- Cost per inspection

## 8. Agent task success

| Metric | Definition |
| --- | --- |
| **Decision attribution rate** | Non-trivial decisions where a named OpenCV metric crossing a named threshold provably caused the branch. The Agentic Vision evidence. |
| Task success rate | Inspections reaching a correct decision *or* a correct escalation |
| Safe-failure rate | Wrong-but-escalated ÷ all wrong outcomes. Higher is better. |
| Unsafe-failure rate | Confidently wrong without escalation. **The metric that matters most.** |
| Remediation trigger rate | Degraded inputs where remediation was attempted |
| Loop termination | Bound respected; no oscillation between actions |

### 8a. Status after Phase 3

Measured on a **deterministic scenario suite** of 12 constructed fixtures, each
declaring its expected terminal state, model invocation, required tools and
forbidden tools before it runs. Every denominator is stated in
`results/phase3/agent_metrics.json`.

| Metric | Value | Denominator |
| --- | --- | --- |
| `TASK_SUCCESS_RATE` | 1.0 | 12 scenarios |
| `DECISION_ATTRIBUTION_RATE` | 1.0 | 33 action-selecting trace steps |
| `UNSAFE_INFERENCE_RATE` | 0.0 | 12 runs |
| `UNNECESSARY_TOOL_CALL_RATE` | 0.0 | 12 runs |
| `BOUNDED_EXECUTION_RATE` | 1.0 | 12 runs |
| `TRACE_COMPLETENESS_RATE` | 1.0 | 12 traces |
| `FAIL_SAFE_RATE` | 1.0 | 5 known-unusable inputs |

**These are not accuracy figures.** They say the agent did what the policy
specifies on cases written down in advance; they say nothing about how often
real photographs fall into each branch. The recovery, escalation-precision and
harm-rate metrics in §9 and §10 require a labelled real-imagery population and
remain outstanding.

The `rejected_remediation` scenario uses a deliberately tightened acceptance
guard, recorded in its own row as a policy variant: the harm guard never fired
on any natural fixture, and contriving an image until it did would have been
worse than exercising the routing honestly.

## 9. Recovery behaviour

- **Recovery rate**: degraded inputs where the single-shot baseline was wrong and
  the agent reached the correct decision after remediation.
- **Recovery cost**: added latency and actions per recovery.
- **Harm rate**: inputs the baseline got *right* that remediation made wrong.
  Reported even if unflattering — recovery is only a gain net of this.

## 10. Escalation behaviour

- Escalation precision and recall against degradation ground truth.
- Unnecessary escalation rate on clean inputs (the usability cost).
- Escalation reason distribution — is any rule dominating or dead?
- Evidence completeness of escalation records.

## 11. Failure taxonomy

Every failure is classified and counted:

| Class | Description |
| --- | --- |
| F1 | Capture failure not detected |
| F2 | Capture failure detected, remediation ineffective |
| F3 | Segmentation failure |
| F4 | Localisation miss (deterioration present, regions absent) |
| F5 | Model error on adequate capture |
| F6 | Perception/model conflict resolved incorrectly |
| F7 | Escalation that should have been an automatic decision |
| F8 | Missing escalation — **unsafe**, tracked separately |
| F9 | Out-of-taxonomy input mishandled |
| F10 | Infrastructure or timeout failure |

## 12. Human-review criteria

Escalation triggers, fixed before measurement so precision/recall are honest:

1. Capture quality below threshold after bounded remediation.
2. Model uncertainty above the escalation threshold.
3. Perception/model conflict.
4. No produce detected, or produce outside the taxonomy.
5. Domain-shift indicator fired.

Thresholds are configuration, recorded in every trace, and selected on a
development split — **never tuned on the reported evaluation set**.

## 13. Reproducibility expectations

- Single seeded entry point reproduces every reported number.
- Pinned dependencies; container digest recorded.
- Evaluation artifacts committed alongside the code that produced them.
- Degradation suite regenerable deterministically from seed and source manifest.
- Manifest and checksum discipline follows the existing research convention.
- Environment, commit SHA and container digest recorded in every result file.
- Reproduction rehearsed from a clean checkout before submission.

## 14. Explicitly excluded metrics

Named so their absence is a stated decision, not an oversight:

| Excluded | Reason |
| --- | --- |
| Pathogen / toxin / contamination detection accuracy | Outside the claim boundary; no ground truth; would be fabrication |
| Internal spoilage or shelf-life prediction | No internal-state data; not a visual quantity |
| Edibility or food-safety accuracy | Explicitly prohibited claim |
| Absolute "freshness percentage" | No calibrated ground-truth scale exists in the data |
| Human-inspector agreement | No annotated inspector panel available |
| Field-deployment or economic-impact figures | No field trial; would be speculation |
| Real-time video FPS on edge hardware | No target device in scope |
| Cross-species generalisation beyond the six classes | No labelled data outside the taxonomy |
