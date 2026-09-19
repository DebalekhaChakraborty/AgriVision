# Phase 6 — Fresh confirmatory end-to-end evaluation

**Status: COMPLETE. Uncommitted.**
Confirmatory evaluation of the frozen deployed system on material no earlier
phase has looked at. No threshold, policy, model or state transition was
changed at any point after the system was frozen.

---

## 0. What this phase was for

Every headline number the project carried into Phase 6 was one of two things: a
deterministic result on constructed fixtures, or a measurement on a data split
that has since been spent. Phase 2c-B opened its held-out groups once. Phase 2d
opened its validation set once. Phase 3b worked on the Phase 2c-B images. Phase
3's scenarios are drawn shapes.

None of that is worthless, and none of it answers the question a judge will
actually ask: *what does the deployed thing do with a photograph it has never
seen?*

So Phase 6 built a third pool, froze the system before opening it, preregistered
what the agent was expected to do, and ran the **deployed AWS service** rather
than a local pipeline.

The result is not uniformly flattering, and it was not made flattering. One
preregistered variant failed almost completely, and the diagnosis is in §7.

---

## 1. The system under evaluation

Recorded in [`frozen_system.json`](../../competition/evaluation/results/phase6/frozen_system.json)
**before** any Phase 6 image was fetched.

| | |
| --- | --- |
| Frozen fingerprint | `97b059be36bc3fbe` |
| Commit | `83a2d9a5ca6b684145fdea4522495d00c7efc8de` (Phase 5) |
| Deployed image digest | `sha256:b57eaf0506a243798ef21423048e1c571606d78243caa13bbf7d11b79d6ce7e7` |
| Deployment operation | `5b391de941c2441683f80ef6989a95e2`, succeeded 2026-09-19T15:19:25Z |
| OpenCV | 5.0.0 local **and** on the service |
| Model | `mobilenet_v3_large-v2expV2-P2-MOBILENETV3-LARGE-004`, SHA-256 `77d8614671…`, verified at start-up |
| ROI policy | `78b1e2151773787a` / lock `15b7fb0908d6d622`, status `PARTIALLY_CALIBRATED_SOME_THRESHOLDS_PROVISIONAL` |
| Foreground fallback | **disabled**, as shipped |
| Budget | 1 remediation, 1 inference, 2 segmentations, 24 steps |

The frozen fingerprint covers the commit, every policy fingerprint, every
component version, the model manifest and the execution budget. It was
regenerated after the Phase 6 tooling was written and came back **identical**,
which is the intended demonstration: adding evaluation code cannot move the
system under evaluation.

The image tag is `:phase4` and the tag is mutable. The **digest** is the
identity; the tag is recorded only to show which pointer was resolved.

---

## 2. Fresh sources, and how independence was enforced

Independence is enforced at three levels, not intended at one.

| Level | Rule | Blocked |
| --- | --- | --- |
| Image | every file ever screened, admitted or rejected | 515 ids |
| Creator | any source group in either prior corpus | 115 groups |
| Category | Commons categories used by neither prior harvest | 28 categories |

The third level does the work people forget. Paging deeper into an
already-harvested category mostly re-surfaces files that were screened and
rejected the first time. Those are licence-clean and creator-clean, and they
still carry a curation judgement someone already made while looking at them.

**Measured overlap with all prior corpora: 0 source groups, 0 image ids,
0 content hashes.**

### Curation

371 licence-clean candidates were reviewed by eye on 11 contact sheets, before
any AgriVision metric was computed on any of them. Kept: exactly one produce
item dominant, whole and uncut, natural backgrounds welcome. Rejected: piles,
crates, stalls, orchards, cut fruit, and photographs where the fruit is
incidental to a scene of something else.

A connected hand of bananas counts as one item. Bananas are sold that way, and
the alternative — refusing every banana photograph that is not a single detached
finger — would have left the fruit unrepresented.

### The orange shortfall, reported rather than closed

Target was 15 per fruit. Apple and banana reached it. Orange did not.

**84 orange candidates across 13 fresh categories yielded 8 that satisfy the
one-primary-item contract.** Permissively licensed orange photography on Commons
is overwhelmingly market piles, cut fruit showing the interior, juice, and
orchard scenes — the interior is the interesting part of an orange, so that is
what people photograph.

The contract was not relaxed to reach 15. Relaxing it would have changed what
Track R measures in order to make a count come out right.

Apple and banana were capped at 15 by sorting kept image ids and taking the
first fifteen — deterministic, independent of appearance and of every metric,
and fixed before the pipeline ran.

**Final pool: 38 images, 38 distinct source groups, 8 licences.**
All permit commercial use and derivatives. No Unknown, no NonCommercial, no
NoDerivatives. Raw bytes are gitignored; only provenance is committed.

---

## 3. Two tracks, never pooled

**Track R** — 38 fresh natural photographs, sent in the format they were
harvested in. Asks what the frozen system does in the natural domain.

**Track C** — 12 of those photographs × 4 controlled variants = 48 scenarios,
sent as PNG. Asks whether the agent routes correctly when the condition is
*known*.

They answer different questions and are never combined into one headline
number. Track R has no ground truth for visual condition; Track C has no claim
to natural prevalence.

### Label boundary

Fruit type was recorded at curation from the source, before any model ran, so
fruit-type accuracy is a real measurement.

**Visible-condition accuracy is not computed and is not computable here.** These
photographs carry no independent condition label. Deriving one from the model's
own output would be circular, and inventing one by eye would be a single
unqualified rater's opinion presented as truth.

---

## 4. Preregistration

Fixed in
[`phase6_expected_actions.json`](../../competition/evaluation/results/phase6/phase6_expected_actions.json),
fingerprint `9ce69ebe042428dc`, before a single variant was executed.

| Variant | Expected first action | Model must run | Expected terminal |
| --- | --- | --- | --- |
| `REFERENCE` | `NONE` | yes | `COMPLETE` |
| `UNDEREXPOSED_RECOVERABLE` | `APPLY_GAMMA` | yes | `COMPLETE` |
| `SEVERE_BLUR` | `REQUEST_RECAPTURE` | **no** | `REQUEST_RECAPTURE` |
| `LOW_CONTRAST_RECOVERABLE` | `APPLY_CLAHE` | yes | `COMPLETE` |

Two denominators were fixed at the same time:

- **Primary** — all 48 scenarios. A base whose reference never produced a valid
  foreground still counts, and its variants count as failures. Perception
  failing before the routing question is reached is a real limitation, not an
  exemption.
- **Secondary** — the 36 scenarios whose base produced a valid foreground on
  `REFERENCE`. Reported *alongside* the primary, never instead of it, because it
  separates "the agent routed wrongly" from "the agent never got far enough to
  route".

Bases were chosen by `sha256(salt:image_id)` within each fruit, salt fixed in
advance. Selection cannot see a pixel or a metric.

---

## 5. Track R — natural domain

Every figure from the **deployed service**. 38/38 returned HTTP 200.

| Metric | Value | Numerator / denominator |
| --- | --- | --- |
| Foreground-valid rate | **76.3%** | 29 / 38 |
| Insufficient visual evidence | 23.7% | 9 / 38 |
| Complete-inspection rate | **73.7%** | 28 / 38 |
| Recapture rate | 26.3% | 10 / 38 |
| Human-action rate | 26.3% | 10 / 38 |
| Remediation attempt rate | 2.6% | 1 / 38 |
| Remediation acceptance rate | 100% | 1 / 1 |
| **Fruit-type accuracy** | **96.4%** | **27 / 28 model invocations** |
| Unsafe inference rate | **0%** | 0 / 38 |
| Trace retrievable | 100% | 38 / 38 |
| Policy fingerprints consistent | yes | 1 distinct set |

**Complete-inspection rate is not accuracy.** It says how often the system
reached a result, not how often the result was right.

The fruit-type denominator is 28, not 38. Ten images were blocked before the
model ran; they are neither right nor wrong, and inflating the denominator with
them would understate the model while inflating the pool.

The single error: an apple predicted banana at confidence **0.233** — the
lowest-confidence prediction in the set. Confidence is UNQUALIFIED for policy
and no threshold is introduced here, but the distributions are reported in
`track_r_results.json` as descriptive domain-shift evidence.

### Segmentation is still the binding limit

The nine foreground failures, by reason: `EXCESSIVE_BORDER_CONTACT` 6,
`MASK_FRAGMENTED` 2, `NO_DOMINANT_COMPONENT` 2.

Worth stating plainly: **76.3% is much better than Phase 3b's 41% overall and
30% on natural scenes**, and no segmentation code changed between them. The
difference is the deployment input contract. Phase 3b measured a corpus
containing market stalls and crates; Track R enforces one primary item. The
contract is doing the work, which is what a scope boundary is for — and it also
means the figure describes captures that obey the contract, not captures in
general.

---

## 6. Track C — routing under known conditions

**These are controlled degradations applied to real images.** Not real
camera-failure prevalence, not phone-camera validation, not the real-world
environmental distribution.

| Metric | Value | Numerator / denominator |
| --- | --- | --- |
| Action-selection accuracy (primary) | **62.5%** | 30 / 48 |
| Action-selection accuracy (secondary) | 75.0% | 27 / 36 |
| Task-success rate | 60.4% | 29 / 48 |
| **Decision-attribution rate** | **100%** | **58 / 58 decisions** |
| **Unsafe-inference rate** | **0%** | 0 / 48 |
| Unnecessary tool-call rate | 0% | 0 / 12 must-not-run scenarios |
| Remediation success rate | 90% | 9 / 10 |
| Remediation harm rate | 10% | 1 / 10 |
| Bounded-execution rate | 100% | 48 / 48 |
| Trace-completeness rate | 100% | 48 / 48 |
| **Fail-safe rate** | **100%** | 12 / 12 `SEVERE_BLUR` |

### Per variant

| Variant | Matched preregistered action |
| --- | --- |
| `SEVERE_BLUR` | **12 / 12** |
| `REFERENCE` | 8 / 12 |
| `LOW_CONTRAST_RECOVERABLE` | 8 / 12 |
| `UNDEREXPOSED_RECOVERABLE` | **2 / 12** |

The safety-critical row is the one that matters most, and it is perfect. On
every severely blurred real photograph the agent asked for a recapture and
**never once ran the condition model**. Detail destroyed by defocus cannot be
recovered by any enhancement, and the system behaves as if it knows that.

---

## 7. The negative finding, and why it is not tuned away

`UNDEREXPOSED_RECOVERABLE` scored **2/12**. That is the honest number and it
stays.

The diagnosis is unambiguous. Measuring where each variant's pixels actually
landed against the calibrated thresholds:

| Landing zone | Bases |
| --- | --- |
| Shadows clipped — detail gone, not recoverable | 7 |
| Above the underexposure floor — nothing to correct | 2 |
| No valid foreground | 3 |
| **Actually in the recoverable band** | **0** |

**Not one of the twelve variants landed in the band the scenario was named
after.** The exposure gain of 0.12 was chosen against a synthetic 256×256
fixture with a mid-grey background. Real photographs already contain deep
shadows, so the same gain drives them straight through the narrow recoverable
band and into clipping.

So the agent was not misrouting. It was refusing images whose shadow detail had
genuinely been destroyed, which is correct. The scenario was mislabelled by its
own construction.

This is classified as `SCENARIO_CONSTRUCTION_DEFECT`, not `QUALITY_FALSE_BLOCK`.
Charging it to the agent would blame the system for reading the pixels
correctly.

The irony is recorded rather than buried: the `LOW_CONTRAST` variant was made
adaptive during Phase 5 *precisely because a fixed factor overshot*, and the
exposure variant was left fixed. The same lesson, not applied twice.

### What was deliberately not done

The variant was **not** rebuilt and rerun as the confirmatory result. Adjusting
evaluation data until it produces the preregistered answer is exactly what
preregistration exists to prevent.

### Exploratory addendum, clearly separated

A separate, non-confirmatory diagnostic
([`exploratory_underexposure.json`](../../competition/evaluation/results/phase6/exploratory_underexposure.json))
swept the exposure gain per base to find one landing in the recoverable band,
searching over the *image* and never over a threshold. Two findings:

1. **The band is reachable on only 3 of 12 real photographs.** For the other
   nine, any gain low enough to cross the floor clips the shadows first.
2. **On all 3 where it is reachable, the agent selected `APPLY_GAMMA`** — 3/3.
   The gamma route works when the condition genuinely exists.

Finding 1 is about the calibrated threshold, not about the agent. The floor sits
at ROI mean luminance `0.0527`, which is very dark; global darkening reaches the
shadow-clipping limit before it reaches that floor on most real photographs. It
belongs on the Phase 7 list, and it is stated here rather than left for someone
else to find.

These numbers are never pooled with Track C.

### The other failures

- **`FOREGROUND_FAILURE` 19** (10 Track C, 9 Track R). The dominant failure
  mode, as in every phase since 2b.
- **`SOURCE_IMAGE_BLOCKING_CONDITION` 1.** One unmodified reference photograph
  carries clipped shadows of its own. The agent refused correctly; the
  preregistration was too optimistic in assuming every fresh photograph is
  usable.
- **`POLICY_MISMATCH` 1.** One low-contrast variant landed just above the
  advisory limit, so the agent did nothing and completed. Adaptive targeting
  missing narrowly on one image.
- **`MODEL_FRUIT_TYPE_ERROR` 1.** The apple/banana confusion above.

### Segmentation as a confounder, per protocol

One case is exactly what §B14 anticipated: on base `WC-41881957d489f4` the
`REFERENCE` foreground was valid, but the `LOW_CONTRAST` variant lost it. The
controlled degradation changed segmentation validity. It is recorded as
`FOREGROUND_FAILURE` and counted as a scenario failure. The expectation was not
rewritten to accommodate it.

---

## 8. Counterfactual pairs — the Agentic Vision evidence

Each base contributes four rows differing only in a controlled visual condition.
Subject, camera and background are held constant.

**10 of 12 bases change their selected action when the condition changes.**
Four distinct first actions across the matrix: `NONE` (9), `APPLY_CLAHE` (8),
`APPLY_GAMMA` (2), `REQUEST_RECAPTURE` (29).

The 2 invariant bases are the ones whose foreground could not be isolated, so
every variant stopped at the same earlier refusal and the routing question was
never reached. They are listed, not dropped.

Because only the pixels differ, the OpenCV evidence is the only thing that could
have changed the action. Combined with **58/58 decision attribution**, every one
of those actions carries the metric it rests on, the threshold it crossed, and
the maturity of that evidence.

This replaces the constructed-fixture examples as the strongest available
Agentic Vision evidence:
[`final_evidence_table.json`](../../competition/evaluation/results/phase6/final_evidence_table.json),
[`counterfactual_matrix.json`](../../competition/evaluation/results/phase6/counterfactual_matrix.json).

---

## 9. Deployed latency

**Client-observed, end-to-end, over the public internet.** Includes TLS, network
transit and image upload. **It is not server compute time.**

| Path | n | Median | p95 |
| --- | --- | --- | --- |
| Track R natural | 38 | 1326 ms | 3114 ms |
| Track C remediation | 10 | 2023 ms | 6265 ms |
| Track C recapture | 29 | 992 ms | 3479 ms |

Substantially slower than Phase 5's 300–400 ms, and the reason is the images:
Phase 5 used small generated fixtures, Track R uses real photographs up to
1920 px. Segmentation and ROI measurement scale with pixels.

Server-side per-stage timings are kept separately in
[`observability_audit.json`](../../competition/evaluation/results/phase6/observability_audit.json);
the model itself is a 22 ms median of that total.

---

## 10. Observability audit

| Check | Result |
| --- | --- |
| Trace persisted and retrievable | **PASS** — 89/89 by run id |
| Run id consistent across response and trace | **PASS** |
| Policy fingerprints match the frozen system | **PASS** — every response |
| No image bytes or paths in responses or traces | **PASS** |
| Unsafe-inference counter inspectable | **PASS** — absent, i.e. zero |
| Structured JSON log emitted to CloudWatch | **NOT VERIFIED** |

The last row is a real gap, recorded as unperformed rather than assumed. The
scoped deployment identity is denied `logs:FilterLogEvents` by design, and
creating a read-only audit policy was not permitted in this environment. What is
established is that the formatter redacts correctly under unit test; what is
**not** established is that the deployed process emitted those records, because
no log line was read back.

Closing it needs a read-only policy granting `logs:FilterLogEvents` on
`/aws/apprunner/agrivision-inspection/*`. Deliberately not done unilaterally: it
widens a deployment identity that was narrowed on purpose one phase ago.

On the counter: the service publishes a counter only once it has been
incremented, so `unsafe_inference_count` is **absent**, which is how zero
presents. That is stated as an absence rather than dressed up as a printed zero,
and it agrees with the measured 0/48 and 0/38.

---

## 11. Limitations

- **38 images and 48 scenarios are small.** Every rate here has a wide interval
  and none is drawn on the charts, because at these counts an interval would be
  wide enough to be useless and decorative enough to look authoritative.
- **One curator, one pass.** Track R membership reflects a single rater's reading
  of the one-primary-item contract.
- **Orange is under-represented**: 8 images against 15 apple and 15 banana. Any
  per-fruit orange figure rests on 8 photographs.
- **Track R measures captures that obey the deployment contract.** It is not a
  measurement of arbitrary user photographs, and the 76.3% foreground rate would
  not survive contact with market stalls.
- **Wikimedia Commons is not the deployment domain.** It is licence-verifiable
  real photography, which is a different and lesser thing than phone captures in
  a kitchen.
- **No visible-condition ground truth exists**, so the model's actual job —
  condition assessment — remains unmeasured on real imagery. Fruit type is a
  proxy for domain fit, not a measure of the product claim.
- **Controlled degradations are not prevalence.** Track C says routing is correct
  when a condition is present; it says nothing about how often it is.
- **CloudWatch emission unverified**, as above.

---

## 12. Reproduction

```bash
# 1. freeze the system - before touching any evaluation data
python -m competition.evaluation.phase6_frozen_system revision.json

# 2. screen, review, harvest Track R  (contact sheets are reviewed by a human)
python -m competition.evaluation.phase6_source_pool screen
python -m competition.evaluation.phase6_source_pool sheets
python -m competition.evaluation.phase6_source_pool harvest

# 3. preregister Track C and build the variants
python -m competition.evaluation.phase6_track_c

# 4. run the DEPLOYED service
python -m competition.evaluation.phase6_run r
python -m competition.evaluation.phase6_run c

# 5. score, audit, illustrate
python -m competition.evaluation.phase6_metrics
python -m competition.evaluation.phase6_evidence
python -m competition.evaluation.phase6_observability
python -m competition.evaluation.phase6_figures

# 6. exploratory addendum - NOT confirmatory
python -m competition.evaluation.phase6_exploratory
```

Steps 2 and 4 need network access; step 2 needs a human at the contact sheets.
Everything else is deterministic given the manifests.

---

## 13. What Phase 6 establishes, and what it does not

**Establishes, on fresh real photographs, on the deployed AWS service:**

- OpenCV 5 evidence changes the subsequent action — 10 of 12 bases, four
  distinct actions, identical policy fingerprints.
- Every recorded decision names the evidence it rests on — 58/58.
- The model is never invoked on a capture the agent blocked — 0/86.
- Severe blur always produces a recapture request and never an inference —
  12/12.
- Fruit-type prediction survives the domain shift — 27/28.

**Does not establish:**

- Visible-condition accuracy on real imagery. No label exists.
- That the recoverable-underexposure route fires on natural captures. The band
  is reachable on 3 of 12 photographs, and the confirmatory variant never
  reached it at all.
- That segmentation is solved. It is the binding limit, still, at 76.3% under a
  contract that already excludes the hard cases.
- Anything about phone captures, which remain uncollected.
