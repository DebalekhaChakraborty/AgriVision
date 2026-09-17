# Phase 1 — OpenCV 5 Capture-Quality Perception

Technical note for the first substantive OpenCV 5 component of the competition
system. Scope is capture-quality assessment only: no produce-condition
classification, no agent controller, no AWS.

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Phase 0 base commit | `ad1e707837ef70863aa3e503b439a225970df15e` |
| OpenCV version | **5.0.0** (`opencv-python==5.0.0.93`) |
| Pipeline version | `phase1-capture-quality-1.0.0` |
| Degradation version | `phase1-degradation-1.0.0` |
| Threshold status | `PROVISIONAL_DEVELOPMENT_ONLY` |
| Tests | 81 passing |

**Claim boundary.** Everything here measures the *photograph*, never the
produce. No metric in this phase indicates freshness, surface deterioration,
safety or edibility.

---

## 1. What was implemented

| Module | Responsibility |
| --- | --- |
| `competition/vision/quality.py` | OpenCV measurement and flag derivation |
| `competition/vision/evidence.py` | Typed, JSON-serialisable evidence contract |
| `competition/vision/config.py` | Threshold policy, separated from measurement |
| `competition/vision/degradation.py` | Deterministic controlled degradation |
| `competition/vision/fixtures.py` | Synthetic, licence-clean test images |
| `competition/vision/__main__.py` | CLI entry point |
| `competition/evaluation/phase1_baseline.py` | Sweep evaluation, plots, latency |

## 2. OpenCV 5 APIs used

| API | Purpose |
| --- | --- |
| `cv2.cvtColor(COLOR_BGR2GRAY)` | Grayscale for the Laplacian |
| `cv2.cvtColor(COLOR_BGR2LAB)` | CIELAB L* for all exposure statistics |
| `cv2.Laplacian(CV_64F)` | Second-derivative response for sharpness |
| `cv2.GaussianBlur` | Controlled defocus degradation; glare falloff |
| `cv2.convertScaleAbs` | Exposure and contrast degradation with saturation |
| `cv2.circle`, `cv2.rectangle` | Synthetic glare and occlusion |
| `cv2.imread`, `cv2.imwrite` | CLI image I/O |

`cv2.ximgproc` is unavailable in this wheel (contrib not installed), so the
design uses core modules only.

## 3. Measured metrics

**Sharpness** — variance of the Laplacian on the grayscale image. The Laplacian
sums second spatial derivatives and responds to intensity transitions; a focused
image has many strong edges and a wide response spread. Direction: higher means
sharper.

**Illumination** — all statistics on the CIELAB L* channel, which approximates
perceptual lightness better than a BGR mean and, unlike HSV's V (a plain channel
maximum), accounts for how the channels combine.

- `mean_luminance`, `median_luminance` — central tendency, normalised 0–1
- `contrast_score` — (p95 − p5) spread of L*, normalised; percentiles so that a
  few extreme pixels do not dominate
- `shadow_clip_fraction`, `highlight_clip_fraction` — fraction of pixels at or
  beyond the clipping bounds, which is what separates "dark" from "lost"

## 4. Evidence contract

`PerceptionEvidence` is a stdlib dataclass — no Pydantic, because that
dependency is not yet verified and the contract should not wait on it.

Design decisions worth noting:

- **Identity by content hash, never by path.** Evidence carries
  `content_sha256`, not a filename. Several committed V2 research result files
  embed absolute paths from an earlier directory layout and went stale when the
  repository moved; a test enforces that this is not repeated.
- **Timing is isolated.** `processing_ms` is wall-clock and legitimately varies,
  so `deterministic_payload()` excludes it. Determinism is asserted on that
  payload, keeping the guarantee honest rather than vacuous.
- **No single opaque score.** Individual components are preserved so later agent
  branching can name *which* measurement caused an action.
- **Policy provenance is recorded.** Every record carries the threshold policy
  fingerprint and status, so a historical decision can be re-derived and records
  produced under different policies are not silently compared.

## 5. Threshold policy

Thresholds live in `config.py`, never inside measurement functions. Raw metrics
are policy-independent; only flags respond to policy — enforced by a test.

Flags: `BLUR_RISK`, `UNDEREXPOSED`, `OVEREXPOSED`, `LOW_CONTRAST`,
`SHADOW_CLIPPING`, `HIGHLIGHT_CLIPPING`, `IMAGE_TOO_SMALL`.

**All thresholds are `PROVISIONAL_DEVELOPMENT_ONLY`.** They are first guesses
chosen to be reasonable on synthetic fixtures. None is calibrated on real
produce photography. Calibration happens in Phase 6 on a development split, and
must never use the frozen research test set.

## 6. Controlled degradation

Measurement apparatus, not training augmentation. Each transform is
parameterised by an explicit level, reproducible from `(image, spec)` alone, and
never mutates its source.

| Family | Levels |
| --- | --- |
| `gaussian_blur` (sigma) | 0, 0.5, 1, 2, 3, 5, 8 |
| `underexpose` (gain) | 1.0, 0.8, 0.6, 0.4, 0.25, 0.15 |
| `overexpose` (gain) | 1.0, 1.3, 1.6, 2.0, 2.6, 3.5 |
| `reduce_contrast` (factor) | 1.0, 0.8, 0.6, 0.4, 0.25, 0.1 |
| `glare` (intensity) | 0, 0.3, 0.5, 0.7, 0.9 |
| `occlude` (area fraction) | 0, 0.1, 0.2, 0.35, 0.5 |

Every ladder starts at its identity level, so each curve has a built-in
undegraded reference. Glare and occlusion placement is seeded and reproducible.

## 7. Test fixtures and licensing

Every fixture is generated from code: checkerboard, gradient, flat fields, and a
synthetic textured object. **No photograph is committed.** The Kaggle source
dataset has an Unknown licence with no redistribution grant and FruitVision is
CC BY-NC-ND 4.0; neither may be committed, and neither is needed to test
capture-quality measurement. No restricted imagery was added in this phase.

## 8. Baseline results

144 records — 4 fixtures × 36 degradation levels. Full data in
`competition/evaluation/results/phase1/`.

Representative response on the `textured_object` fixture:

**Blur → sharpness** (monotone until the noise floor)

| sigma | Laplacian variance | flags |
| --- | --- | --- |
| 0.0 | 1036.19 | — |
| 0.5 | 276.46 | — |
| 1.0 | 10.04 | BLUR_RISK |
| 2.0 | 1.64 | BLUR_RISK |
| 5.0 | 0.68 | BLUR_RISK |
| 8.0 | 0.83 | BLUR_RISK |

**Overexposure → luminance and highlight clipping** (monotone)

| gain | mean L* | highlight clip | flags |
| --- | --- | --- | --- |
| 1.0 | 0.48 | 0.000 | — |
| 1.6 | 0.71 | 0.000 | — |
| 2.0 | 0.82 | 0.039 | OVEREXPOSED |
| 2.6 | 0.95 | 0.179 | OVEREXPOSED, LOW_CONTRAST, HIGHLIGHT_CLIPPING |
| 3.5 | 1.00 | 0.908 | OVEREXPOSED, LOW_CONTRAST, HIGHLIGHT_CLIPPING |

No accuracy, precision or recall is reported. No ground-truth condition labels
exist at this phase, so any such number would be fabricated.

`phase1_sweep.json` and `phase1_sweep.csv` are **byte-reproducible**: per-record
wall-clock timing is deliberately excluded, since a single sample is
statistically meaningless and would make these files churn on every run. A diff
in them therefore signals a real change in measurement. `phase1_latency.json` is
the only intentionally non-deterministic artifact.

## 9. Findings and limitations

### 9.1 Sharpness is not exposure-invariant — the most significant finding

With **zero blur applied**, changing only exposure moves the sharpness metric
substantially:

| exposure gain | Laplacian variance | BLUR_RISK |
| --- | --- | --- |
| 0.15 | 23.70 | **raised (false positive)** |
| 0.25 | 66.61 | **raised (false positive)** |
| 0.40 | 166.15 | no |
| 1.00 | 1036.19 | no |
| 1.60 | 1850.00 | no |
| 2.60 | 187.60 | no |

Laplacian variance scales approximately with the **square of signal amplitude**:
1036.19 × 0.4² = 165.8, against a measured 166.15. Above gain ≈1.6 the trend
reverses as clipping flattens gradients.

Two consequences, both material:

1. **A dark capture can be falsely flagged as blurred.** Observed here at gains
   ≤ 0.25 on an image with no blur at all.
2. **A bright capture can mask real blur** through inflated variance.

This is a strong, evidence-backed argument for the agentic ordering the
blueprint proposes: a single-pass assessment is not sufficient, because the
metrics are coupled. Exposure must be corrected and sharpness re-measured before
a blur verdict can be trusted — precisely the
`assess → enhance → re-assess` loop of Phase 3, now motivated by measurement
rather than by assertion.

Mitigation options for Phase 2/3, none yet implemented: normalise variance by
local contrast, gate the blur verdict on exposure being within band, or measure
sharpness only after enhancement.

### 9.2 Sharpness saturates at a noise floor

Below roughly 1.0 the variance bottoms out and can drift *upward* with larger
kernels (0.68 at sigma 5, 0.83 at sigma 8, 0.93 at sigma 12) as border handling
and uint8 quantisation dominate. Monotonicity is therefore asserted only above
`sharpness_noise_floor` (2.0); below it the metric means "no detail" and nothing
finer. A dedicated test pins this behaviour rather than hiding it.

### 9.3 Shadow clipping is not a synonym for "dim"

Multiplicative underexposure must be drastic before pixels reach the L* shadow
floor — around gain 0.05, not 0.15. At gain 0.15 the fixture is clearly too dark
and is flagged `UNDEREXPOSED`, but nothing is clipped and detail remains
recoverable. The distinction matters for later remediation: a dim-but-unclipped
capture can be rescued by CLAHE, a crushed one cannot.

### 9.4 Contrast responds earlier than mean luminance

On the underexposure ladder, `LOW_CONTRAST` fires at gain 0.6 while
`UNDEREXPOSED` only fires at 0.4, because darkening compresses the L* spread
before it moves the mean past its threshold. `LOW_CONTRAST` is therefore the
more sensitive early indicator of underexposure in this configuration. Whether
that is desirable is a calibration question for Phase 6.

### 9.5 Other known limitations

- Sharpness is scale- and content-dependent; a flat field scores as "blurred"
  because it genuinely has no high-frequency content. Comparisons are only safe
  within an image across a sweep.
- Defocus and motion blur are not distinguished.
- Noise inflates the sharpness metric.
- Clipping is measured on L*, so a single saturated colour channel in an
  otherwise mid-toned pixel is not counted.
- Synthetic degradation is a proxy for real capture failure. Self-captured real
  degraded images are needed to confirm the conclusions transfer.
- All fixtures are synthetic; no behaviour on real produce photography has been
  measured yet.

## 10. Latency

Local CPU baseline only. **Not an AWS measurement** — no container, no cold
start, no network.

| Metric | Value |
| --- | --- |
| Image size | 256×256 |
| Samples | 200 (after 5 discarded warm-up runs) |
| Median | **3.41 ms** |
| Mean | 3.45 ms |
| p95 | **3.66 ms** |
| p99 | 4.02 ms |
| Runtime | Python 3.11.2, Linux x86_64, OpenCV 5.0.0 |

Warm-up runs are discarded because the first assessment absorbs OpenCV and NumPy
lazy initialisation and is roughly two orders of magnitude slower (~220 ms)
than steady state. That cold-path cost is itself relevant to the Phase 4
App Runner versus Lambda decision.

## 11. Tests

81 tests, all passing, covering the nine required behaviours:

| # | Requirement | Where |
| --- | --- | --- |
| 1 | Identical input → identical evidence | `test_evidence.py` |
| 2 | JSON serialisation succeeds | `test_evidence.py` |
| 3 | OpenCV major version is 5 | `test_quality.py` |
| 4 | Increasing blur degrades sharpness | `test_quality.py` |
| 5 | Underexposure moves illumination metrics | `test_quality.py` |
| 6 | Overexposure moves illumination metrics | `test_quality.py` |
| 7 | Invalid input fails safely | `test_quality.py` |
| 8 | No source image is overwritten | `test_degradation.py` |
| 9 | No absolute filesystem path in output | `test_evidence.py` |

Plus coverage of policy/measurement separation, seed reproducibility, transform
parameter validation, and the two documented non-monotonic behaviours.

## 12. Reproduction

```bash
python3 -m venv .venv-competition
.venv-competition/bin/pip install -r requirements-competition.txt

# tests
.venv-competition/bin/python -m pytest tests/competition -q

# baseline sweep, plots and latency
.venv-competition/bin/python -m competition.evaluation.phase1_baseline

# single image
.venv-competition/bin/python -m competition.vision path/to/image.jpg
.venv-competition/bin/python -m competition.vision path/to/image.jpg --no-timing
```

`--no-timing` omits `processing_ms`, producing byte-identical output across runs.

A dedicated `.venv-competition` is used rather than `.venv-v2` because the
latter produced Experiments 001–015 and is pinned by `requirements-v2.txt`;
installing competition tooling into it would mutate the environment behind
published research results.

## 13. Agentic readiness

Phase 1 implements no controller, but the perception functions are already
tool-shaped for Phase 3:

- `assess_capture_quality(image, policy)` takes a decoded array, not a path
- no file I/O, no global state, never mutates its input
- returns typed, serialisable evidence suitable for a trace record
- flag derivation is a separate pure function over raw metrics

which supports the intended sequence without restructuring:

```
assess_capture_quality(image)
        -> policy decision
        -> enhance_capture(image, parameters)
        -> assess_capture_quality(enhanced)
        -> inspect_surface(...) OR request_human_review(...)
```

Finding 9.1 makes that second assessment a measured necessity rather than a
design preference.

## 14. Not done in this phase

Produce-condition classification, image enhancement actions, segmentation, ROI
and anomaly localisation, the agent controller, trace storage, AWS deployment,
Bedrock, and UI. Thresholds remain uncalibrated against real photography.
