# Competition Demo Imagery Plan

Planning document. **No imagery is collected or committed by this plan**, and
nothing here is a task for the current phase.

## Why a separate plan is needed

Every dataset the research programme uses is unsuitable as competition demo
imagery:

| Dataset | Licence | Demo use |
| --- | --- | --- |
| Frozen V1 split (Kaggle `sriramr/...`) | **Unknown** — no redistribution grant found | **No.** Cannot be shown in a public demo, a video, or a judge-accessible archive. |
| Sultana 2022 | CC BY 4.0 | Possible with attribution, but it is someone else's photography of someone else's produce. |
| FruitVision (Mendeley v2) | **CC BY-NC-ND 4.0** | **No.** NonCommercial *and* NoDerivatives; annotated or overlaid versions are derivatives. |

A competition entry involves a public endpoint, a ≤5-minute video and a
judge-accessible repository. All three display images. Local research evaluation
that emits only numbers is a different activity from publishing pictures, and
the licences above do not support the second.

**Recommended path: self-captured photography.** It removes every redistribution
question, and it is strictly better evidence — the whole system is built around
degraded real-world captures, which is precisely what a curated dataset does not
contain.

## Capture set required

Roughly 20–30 items across the three supported fruit types (apple, banana,
orange), each photographed under several conditions. The ontology supports no
other produce, so nothing outside those three should be captured for
condition evaluation.

| Condition | Purpose |
| --- | --- |
| Normal indoor lighting | Baseline; the gate should pass these |
| Dim / underexposed | Gamma remediation path |
| Overexposed | Darkening path, highlight clipping |
| Hard specular glare | Glare handling, future segmentation |
| Handheld motion blur | Recapture path — blur is not remediable |
| Defocus blur | Recapture path |
| Cluttered background | Foreground isolation, future segmentation |
| Partial occlusion | Robustness |
| Mixed fruit in frame | Known limitation: single-label model |
| Non-produce object | Out-of-taxonomy refusal |
| Visibly deteriorated produce | Condition classes need real positives |

Each capture should be recorded with camera/phone model, approximate lighting,
distance, and whether the degradation was deliberate.

## Why this also fixes a measured problem

Phase 2 found the capture gate rejects **96.7%** of the research photographs
because whole-image clipping statistics are dominated by bright backgrounds
rather than by the produce
([PHASE2_MODEL_SELECTION.md](PHASE2_MODEL_SELECTION.md) §12). Threshold
calibration needs a development set that looks like the deployment domain, and
the research dataset is not that domain. Self-captured imagery is needed for
honest calibration, not only for licensing.

## Handling rules

- Demo and calibration imagery is committed only if the capturer holds the
  rights and agrees to redistribution under a stated licence.
- Capture no identifiable people; produce and background only.
- Record a `CAPTURE_PROVENANCE.md` alongside: who captured, when, device,
  licence granted.
- Keep the split honest — calibration images must not later be reported as an
  evaluation set.
- Research imagery stays local, under existing terms, for numeric evaluation
  only.

## Superseded by the Phase 2c protocol

The condition taxonomy sketched above is now **locked and expanded** in
[PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md](PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md),
which is the authoritative document. Three changes were made and recorded there:
"visibly deteriorated produce" became an item attribute rather than a capture
condition, "mixed fruit" and "non-produce object" became scene cases, and blur
and darkness were split into mild and severe so calibration can locate the
boundary between remediable and recapture-required.

This document remains valid as the statement of *why* self-captured imagery is
needed; the protocol document states *how*, and supersedes the taxonomy above.

## Superseded again by Phase 2c-B — three tiers, not one

[Phase 2c-B](PHASE2C_LICENSED_CALIBRATION.md) replaced the single-source
assumption behind this plan. Imagery now comes in three tiers that are never
mixed, because they support different claims:

| Tier | What it is | Status | What it may support |
| --- | --- | --- | --- |
| **Licensed real** | 92 Commons photographs, per-file verified licence, curated by eye | **primary calibration path — done** | behaviour on real imagery |
| **Controlled degradation** | 1,840 characterised transformations of those photographs | **done** | behaviour under a known perturbation |
| **Synthetic / generated** | 12 procedurally composed stress scenes, containing no fruit | **done, exploratory only** | stress probes, nothing else |
| **Self-captured** | phone captures of real fruit | **optional, not started** | camera-domain transfer |

The schema enforces the separation rather than relying on discipline: a
generated record cannot carry a real-image licence or enter a real-image track,
and a derived sample cannot exist without a `LICENSED_REAL` parent.

## Demo imagery — what may actually be shown

Every corpus image is permissively licensed, so public display is permitted.
**62 of 92 carry share-alike** and 82 of 92 require attribution, so any image
shown in the video, on Devpost or in the demo needs its creator, licence and a
link — and an *adapted* share-alike image obliges the adaptation to be offered
under CC BY-SA too.

That obligation is why the demo should not be built around third-party imagery
by default. The shortlist in
[PHASE2C_LICENSED_CALIBRATION.md](PHASE2C_LICENSED_CALIBRATION.md) records which
images are cleanest for this — CC0 and public-domain first, since they carry
neither obligation — but self-captured photographs remain preferable for the
final live demo for exactly the reason this document was written: they are ours,
they are the deployment domain, and they need no attribution slate.

## Status

**Licensed corpus: built and calibrated.** **Self-capture: optional and not
started**, at the reduced scale of 3–6 fruits described in
[PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md](PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md).
It blocks nothing.
