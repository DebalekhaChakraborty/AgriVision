# Phase 1b — Evidence-Driven Capture Remediation

The first explicit **perception → decision → action → re-perception** cycle in
AgriVision, with a structured causal trace.

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Phase 1 base commit | `5248f905b1d67ca9b9cca7c6b93209e93f8ec341` |
| OpenCV version | **5.0.0** (`opencv-python==5.0.0.93`) |
| Enhancement version | `phase1b-enhancement-1.0.0` |
| Policy version | `phase1b-remediation-policy-1.0.0` |
| Trace version | `phase1b-trace-1.0.0` |
| Policy status | `PROVISIONAL_DEVELOPMENT_ONLY` |
| Max automated attempts | **1** |
| Tests | 198 passing (81 Phase 1 + 117 Phase 1b) |

> **Scope statement.** Phase 1b establishes explicit visual tools, an
> evidence-driven deterministic decision, an action, re-perception, and a
> structured causal trace. That is groundwork for Agentic Vision — it is **not**
> a claim that AgriVision satisfies the Agentic Vision Award requirement. That
> claim requires the full inspection orchestrator and a judge-visible trace,
> evaluated against the competition criteria, and is not made here.

**Claim boundary.** Everything in this phase concerns the *photograph*. Nothing
here assesses produce freshness, surface deterioration, safety or edibility.

---

## 1. Empirical motivation

Phase 1 measured that the Laplacian variance sharpness metric is not
exposure-invariant: with **no blur applied**, variance scales with roughly the
square of exposure gain. A dark capture is therefore falsely flagged
`BLUR_RISK`, and a bright one can mask genuine blur.

The consequence is structural, not cosmetic: **a single-pass sharpness verdict
cannot be trusted while illumination evidence says the capture itself is
degraded.** Either the system corrects exposure and re-measures, or its blur
verdict is unreliable exactly when capture conditions are poor — which is
precisely when an inspection system is most likely to be wrong.

Phase 1b addresses that experimentally.

## 2. Architecture

```
assess_capture_quality(image)            [OpenCV]
            |
            v
decide_capture_remediation(evidence)     [deterministic policy]
            |
   +--------+--------+-------------------+
   |        |        |                   |
  NONE   APPLY_    APPLY_        REQUEST_RECAPTURE /
         GAMMA     CLAHE         REQUEST_HUMAN_REVIEW
            |        |                   |
            +---+----+                   v
                |                   (escalate, original canonical)
                v
   enhance_capture(image, decision)      [OpenCV]
                |
                v
   assess_capture_quality(enhanced)      [OpenCV, re-perception]
                |
                v
   compare_capture_quality(before, after)
                |
      +---------+---------+
      |                   |
 ACCEPT_REMEDIATION   REJECT_REMEDIATION
      |                   |
 remediated is        original stays
 canonical            canonical
```

Each box is a separately callable, independently testable function. Nothing is
hidden inside a single nested-conditional routine, so a future orchestrator can
drive the parts directly.

| Module | Responsibility |
| --- | --- |
| `competition/vision/enhancement.py` | `apply_gamma_correction`, `apply_clahe`, gamma derivation |
| `competition/agent/actions.py` | Action enum, reason codes, decision and comparison records |
| `competition/agent/policy.py` | `decide_capture_remediation`, `compare_capture_quality` |
| `competition/agent/trace.py` | Structured trace steps |
| `competition/agent/remediation.py` | Bounded cycle, dispatcher, `RemediationResult` |
| `competition/evaluation/phase1b_remediation.py` | Experiments and metrics |

## 3. OpenCV methods

New APIs beyond Phase 1: `cv2.split`, `cv2.merge`, `cv2.LUT`,
`cv2.createCLAHE(...).apply`, `cv2.COLOR_LAB2BGR`.

**Gamma correction.** Applied to the L\* channel through a precomputed 256-entry
LUT: exact, deterministic, and far cheaper than per-pixel exponentiation.

The gamma value is derived, not guessed. For normalised intensity the power law
is `out = in ** gamma`, so requiring `target = current ** gamma` gives

```
gamma = log(target) / log(current)
```

clamped to the policy range `[0.25, 3.0]`. A test asserts the round trip:
`0.2 ** gamma == 0.45`.

**CLAHE.** `createCLAHE(clipLimit, tileGridSize)` applied to L\* only, then
merged back. Clip limit and tile grid are policy configuration, not buried
constants.

**Both operate on luminance only.** Equalising B, G and R independently would
shift colour balance and corrupt the colour analysis planned for later phases. A
test asserts the a\* and b\* channels are materially unchanged.

**Neither restores clipped information.** A pixel at 0 or 255 has lost its
detail irreversibly; these operations only redistribute what remains. Refusing
to act in that case is the policy's job, not the operation's.

## 4. Policy

Rules are evaluated in severity order so an unrecoverable condition is never
masked by a recoverable one.

| Order | Condition | Action | Rationale |
| --- | --- | --- | --- |
| 1 | `IMAGE_TOO_SMALL` | `REQUEST_RECAPTURE` | No operation adds resolution |
| 2 | shadow clip > 0.10 | `REQUEST_RECAPTURE` | Detail is gone; brightening would fabricate |
| 3 | highlight clip > 0.10 | `REQUEST_RECAPTURE` | Blown regions cannot be recovered |
| 4 | `UNDEREXPOSED`/`OVEREXPOSED`, clipping acceptable | `APPLY_GAMMA` | Tone is recoverable |
| 5 | `LOW_CONTRAST`, exposure acceptable | `APPLY_CLAHE` | Tonal range is compressible |
| 6 | `BLUR_RISK`, illumination in band | `REQUEST_RECAPTURE` | **Phase 1 finding in policy form**: with exposure healthy, low sharpness is genuine blur, and no tone operation removes it |
| 7 | otherwise | `NONE` | Nothing to remediate |

Actions are a **closed enum**. The dispatcher refuses anything that is not a
`RemediationAction` member, so a free-form string cannot cause an image
operation to run — tested directly. **No language model is involved in any
control decision.**

All thresholds are `PROVISIONAL_DEVELOPMENT_ONLY`, chosen conservatively from
Phase 1 synthetic measurements and not calibrated on real photography.

## 5. Acceptance criteria and the harm guard

Success is multi-metric by construction. Deciding from "sharpness increased"
alone would recreate the exact coupling Phase 1 measured — every brightening
would look like a win. **Sharpness is recorded in the comparison but never
decides the verdict**, and a test enforces this.

Accepted only when **both** hold:

1. **The target criterion is met**, by either of two independent routes:
   * the remediated capture **reached the acceptable band** —
     `is_within_target_band(value, lower, upper, tolerance)` for luminance, or
     contrast at or above the acceptable floor for CLAHE; or
   * it **improved by at least the required margin** (0.02).

   Band membership is a tolerance-based test, never a comparison against an
   exact distance of `0.0`: floating-point equality is not a sound way to
   express "landed in the band". The comparison record stores which route
   succeeded (`reached_target_band`, `improved_by_margin`) so a stored decision
   shows *why* the target was considered met.
2. **No guardrail is violated:** shadow clipping and highlight clipping each
   increase by no more than 0.02, and contrast falls by no more than 0.02. The
   band route never overrides the harm guard — tested directly.

**Harm guard.** If a guardrail is violated, or the target fails to improve, the
remediation is rejected, the **original capture remains canonical**, and the
result records `remediation_attempted=true`, `remediation_accepted=false` with a
reason code. Rejection always escalates.

**Bounded loop.** `MAX_AUTOMATED_ATTEMPTS = 1`. If one remediation cannot make
the capture suitable, the result escalates rather than iterating. Repeated
enhance-and-recheck is both a runaway risk and a way to nudge metrics until a
bad image passes.

## 6. Trace

Every cycle produces a structured, JSON-serialisable trace over a closed tool
vocabulary. Images are referenced by content hash; **no filesystem path is ever
recorded**, enforced by test.

Observed sequence for an automated remediation:

```
1  assess_capture_quality       flags [BLUR_RISK, UNDEREXPOSED, LOW_CONTRAST]
                                mean_luminance 0.101, laplacian_variance 66.6
2  decide_capture_remediation   APPLY_GAMMA, UNDEREXPOSED_RECOVERABLE, gamma 0.348
3  apply_gamma_correction       L* only
4  assess_capture_quality       flags [LOW_CONTRAST]
                                mean_luminance 0.441, laplacian_variance 159.2
5  compare_capture_quality      ACCEPT_REMEDIATION, target luminance_distance
6  finalise                     remediated canonical; residual flags -> review
```

For an escalation the trace is three steps: assess, decide, finalise. A worked
example is committed at
`competition/evaluation/results/phase1b/phase1b_example_trace.json`.

## 7. Experiment A — exposure/sharpness coupling

One source, fixed true spatial detail, **no blur applied**, across an exposure
ladder. Any variation in the sharpness metric is measurement artefact.

| gain | action | accepted | Laplacian before | canonical | flags before → after |
| --- | --- | --- | --- | --- | --- |
| 0.05 | REQUEST_RECAPTURE | — | 2.8 | 2.8 | BLUR, UNDER, LOW_C, SHADOW_CLIP → unchanged |
| 0.15 | APPLY_GAMMA | yes | 23.7 | 128.6 | BLUR, UNDER, LOW_C → LOW_C |
| 0.25 | APPLY_GAMMA | yes | 66.6 | 159.2 | BLUR, UNDER, LOW_C → LOW_C |
| 0.40 | APPLY_GAMMA | yes | 166.2 | 217.0 | UNDER, LOW_C → LOW_C |
| 0.60 | APPLY_CLAHE | yes | 374.6 | **2886.4** | LOW_C → none |
| 1.00 | NONE | — | 1036.2 | 1036.2 | none |
| 1.60 | NONE | — | 1850.0 | 1850.0 | none |
| 2.00 | APPLY_GAMMA | yes | 420.1 | **2700.1** | OVER → none |
| 2.60 | REQUEST_RECAPTURE | — | 187.6 | 187.6 | OVER, LOW_C, HIGHLIGHT_CLIP → unchanged |
| 3.50 | REQUEST_RECAPTURE | — | 230.9 | 230.9 | OVER, LOW_C, HIGHLIGHT_CLIP → unchanged |

**False `BLUR_RISK` flags: 3 before → 1 after.** The remaining one is the
genuinely unrecoverable gain-0.05 capture, which the policy correctly refuses to
remediate.

### Exposure sensitivity of the sharpness metric

Measured as **coefficient of variation** (standard deviation ÷ mean) of
Laplacian variance across the ladder. CV is chosen because Laplacian variance
has arbitrary units whose magnitude depends on content and on exposure itself; a
dimensionless ratio stays comparable when remediation shifts the mean, which raw
standard deviation or range would not.

| Subset | Before | Canonical | Direction |
| --- | --- | --- | --- |
| Automated or clean (n=7) | 1.090 | **0.869** | improved 20% |
| Accepted remediations only (n=5) | 0.762 | **1.057** | **worsened 39%** |
| All levels (n=10) | 1.264 | 1.139 | improved 10% |

**These results must not be overstated. Exposure remediation did not make the
sharpness metric exposure-invariant, and on the accepted subset it made
dispersion worse.**

The cause is visible in the table: CLAHE at gain 0.60 raised Laplacian variance
7.7× (374.6 → 2886.4) and strong gamma at 2.00 raised it 6.4× (420.1 → 2700.1).
CLAHE amplifies local contrast, which is exactly what the Laplacian responds to.
So remediation removes the *darkness* confound and introduces a
*gradient-amplification* confound in its place.

The defensible claim is therefore the narrow one, and it is the one B8
anticipated:

> Illumination remediation reduced a confounding factor in the sharpness
> **verdict**.

Not "enhancement fixed blur", and not "sharpness is now exposure-invariant".

## 8. Experiment B — blur × exposure factorial

4 blur levels × 5 exposure gains. Ground truth per blur level is the
`BLUR_RISK` verdict at reference exposure (gain 1.0), where sharpness is
unconfounded. The reference column is excluded from scoring as true by
construction, leaving 16 cells.

Reference verdicts: σ=0.0 → not blurred; σ=1.0, 2.0, 4.0 → blurred.

| Metric | Before | Canonical |
| --- | --- | --- |
| Blur-verdict agreement with reference | 15/16 = **0.938** | 16/16 = **1.000** |

Exactly one disagreement existed before remediation:

| σ | gain | before verdict | truth | action | after verdict |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.25 | BLUR_RISK | not blurred | APPLY_GAMMA | correctly not blurred |

A sharp image, darkened, was misjudged as blurred; exposure remediation
corrected the verdict. That is the mechanism this phase set out to demonstrate.

**Sample-size caveat, stated plainly:** the improvement rests on **a single
corrected cell**. "100% agreement" is not a strong statistical claim on 16
synthetic cells with one correction. It is directionally consistent with the
Phase 1 mechanism, and nothing more. No policy parameter was tuned on this grid.

Notably, no darkening produced a *false negative* — no genuinely blurred image
was made to look sharp — because the affected gains reduce variance rather than
inflate it.

## 9. Engineering metrics

Across all 30 cycles run by the two experiments:

| Metric | Value |
| --- | --- |
| Cycles | 30 |
| Remediation attempt rate | 0.433 (13/30) |
| Remediation acceptance rate | **1.000** (13/13) |
| Remediation rejection rate | **0.000** (0/13) |
| Escalation rate | **0.800** (24/30) |
| Harm rate | **0.000** (0/13) |

These are the corrected rates, after the acceptance defect in §10.1 was fixed.
The pre-fix run reported acceptance 0.692 and rejection 0.308; those figures
were artefacts of the defect and are not carried as results.

The escalation rate is **unchanged at 0.800, but its composition changed**, and
that matters more than the headline. Before the fix, 24 escalations were 13
recapture requests plus 11 cases that were either spuriously rejected or
accepted with residual flags. After the fix they are 13 recapture requests plus
11 **accepted** remediations that still escalate because a residual
`LOW_CONTRAST` flag remains and the single-attempt bound forbids chaining a
second action. The same number of captures reach a human; the difference is that
four of them now do so carrying a correctly remediated image rather than a
spurious rejection.

Action distribution: `APPLY_GAMMA` 12, `REQUEST_RECAPTURE` 13, `NONE` 4,
`APPLY_CLAHE` 1, `REQUEST_HUMAN_REVIEW` 0.

### Harm rate definition

> The fraction of **attempted** remediations rejected because an independent
> critical metric — shadow clipping, highlight clipping, or contrast — worsened
> beyond its guardrail.

Because the harm guard rejects these before they become canonical, **realised
harm to the output is zero by construction**. The metric measures how often
enhancement *would have* damaged the capture, not how often it did.

At 0.000 it is currently **uninformative**, and after the acceptance fix there
are no rejections of any kind on this set, so neither the harm guard nor the
ineffectiveness path is exercised by the evaluation at all. Both are exercised
by dedicated unit tests instead. The metric is defined now, before real
photography can make its definition convenient, and it should be treated as
unmeasured rather than as evidence of safety.

## 10. Findings and limitations

### 10.1 Acceptance criterion demanded more improvement than was achievable (fixed)

**A structural policy defect, found by this evaluation and since corrected.**
Recorded because the mechanism generalises to any threshold expressed as an
absolute improvement margin.

In the first run, all four rejections occurred at exposure gain 0.5, and all
were spurious:

```
before mean L*      0.2370   (band floor 0.25, so distance 0.0130)
after  mean L*      0.4429   (distance 0.0000 - a complete correction)
improvement         0.0130
required margin     0.0200   -> REJECTED as REMEDIATION_INEFFECTIVE
```

The remediation worked perfectly and was still rejected, because
`min_luminance_distance_improvement` (0.02) exceeded the maximum improvement
available — the initial distance itself, 0.013. **For any capture starting
within the margin of the band, remediation was guaranteed to be rejected even
when it fully succeeded.**

The failure was safe (the original was retained and the case escalated) but it
distorted the acceptance and rejection rates, and marginal underexposure could
never be auto-accepted.

**The fix** expresses the policy as two independent routes to target success —
reaching the band, or improving by the margin — rather than the margin alone:

```python
target_improved = reached_band or improved_enough
```

with `reached_band` computed by `is_within_target_band(value, lower, upper,
tolerance)`. A tolerance-based membership test is used deliberately in place of
comparing a distance against exact `0.0`; floating-point equality is not a sound
way to state a policy. Guardrails are unchanged and still apply: the band route
never overrides the harm guard.

**Effect on results.** Acceptance rose from 0.692 to 1.000 and rejections fell
from 4 to 0. Experiment A's exposure ladder contains no gain-0.5 level, so its
sensitivity figures and the false-`BLUR_RISK` count are unaffected; the
factorial's agreement rates are likewise unchanged, because the affected cells
were already being judged on their pre-remediation verdict. The corrected rates
are the ones reported in §9; the pre-fix rates are not carried as results.

Six regression tests pin the corrected semantics, including the marginal case,
a small improvement that stays outside the band (still rejected), and a
band-reaching remediation that violates a harm guard (still rejected).

### 10.2 Remediation introduces a new confound

CLAHE and strong gamma amplify gradients and therefore inflate Laplacian
variance (up to 7.7× here). Sharpness measured on a CLAHE-remediated image is
not comparable with sharpness measured on an untouched one. Until this is
addressed, a blur verdict on a remediated capture should be treated as less
trustworthy, not more. Candidate directions: normalise variance by local
contrast, or record the enhancement in evidence so downstream consumers can
discount it.

### 10.3 Escalation rate is high, by design of the bound

0.800 of cycles escalate. Two causes: 13 captures were genuinely unrecoverable
or genuinely blurred (correct escalations), and accepted gamma remediations
still escalate when a residual flag such as `LOW_CONTRAST` remains, because the
single-attempt bound forbids chaining a second action. A future orchestrator
permitted to chain gamma → CLAHE would reduce this materially. The current
behaviour is conservative and safe, not optimal.

### 10.4 Other limitations

- All fixtures are synthetic; nothing has been measured on real produce
  photography.
- Thresholds remain uncalibrated and provisional.
- `REQUEST_HUMAN_REVIEW` is defined but never selected by the current policy —
  every escalation currently routes to `REQUEST_RECAPTURE`.
- CLAHE was selected only once in the evaluation, so its behaviour is far less
  characterised than gamma's.
- Multiplicative exposure degradation is a proxy for real capture failure.
- The blur/exposure grid is small; the agreement result rests on one cell.

## 11. Tests

198 passing (81 Phase 1, unchanged, plus 117 new). All seventeen required
behaviours are covered:

| # | Requirement | File |
| --- | --- | --- |
| 1 | Gamma deterministic | `test_enhancement.py` |
| 2 | CLAHE deterministic | `test_enhancement.py` |
| 3 | Source never mutated | `test_enhancement.py` |
| 4 | Luminance-only round trip yields valid BGR | `test_enhancement.py` |
| 5 | Decision deterministic | `test_policy.py` |
| 6 | Severe clipping escalates, no fake recovery | `test_policy.py` |
| 7 | One attempt maximum | `test_remediation.py` |
| 8 | Failed remediation preserves original | `test_remediation.py` |
| 9 | Accepted remediation records before + after | `test_remediation.py` |
| 10 | Trace records causal order | `test_remediation.py` |
| 11 | Trace JSON-serialisable | `test_trace.py` |
| 12 | No filesystem paths | `test_remediation.py` |
| 13 | Free-form action cannot execute | `test_remediation.py` |
| 14 | Underexposure triggers remediation | `test_policy.py` |
| 15 | Healthy fixture yields NONE | `test_policy.py` |
| 16 | Harmful enhancement rejected | `test_policy.py` |
| 17 | No research/V1/V2 import | `test_remediation.py` |

Plus six regression tests for the §10.1 acceptance fix, a tolerance test for
`is_within_target_band`, and an executable form of the phase thesis,
`test_exposure_remediation_can_clear_a_false_blur_flag`.

## 12. Reproduction

```bash
python3 -m venv .venv-competition
.venv-competition/bin/pip install -r requirements-competition.txt

.venv-competition/bin/python -m pytest tests/competition -q
.venv-competition/bin/python -m competition.evaluation.phase1_baseline
.venv-competition/bin/python -m competition.evaluation.phase1b_remediation
```

Outputs land in `competition/evaluation/results/phase1b/`:
`phase1b_results.json`, two CSVs, `phase1b_example_trace.json`, and three plots.
All artifacts exclude per-record timing and are byte-reproducible.

## 13. Not done in this phase

Condition classification, segmentation, ROI and anomaly localisation, the full
inspection orchestrator, multi-step or chained remediation, trace persistence,
AWS deployment, Bedrock, and UI. No Agentic Vision Award claim is made.
