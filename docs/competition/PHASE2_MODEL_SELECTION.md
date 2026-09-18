# Phase 2 — Condition-Model Selection and Integration

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Phase 1b base commit | `9983b49f669cf5b1db94d53456a064988ab38a06` |
| OpenCV version | **5.0.0** (`opencv-python==5.0.0.93`) |
| Selected runtime model | **MobileNetV3-Large (V2 Exp 004) → ONNX → `cv2.dnn`** |
| Serving dependencies | `opencv-python`, `numpy` — **no torch** |
| Tests | 259 passing |

> **Scope.** Phase 2 selects and integrates a condition model behind the capture
> gate. It does **not** implement surface segmentation or anomaly localisation,
> and it does **not** implement model-versus-perception conflict reasoning. No
> Agentic Vision Award claim is made.

**Claim boundary.** The model predicts *visible surface condition* across three
fruit types. It makes no food-safety, edibility, contamination or
internal-spoilage claim, and the vocabulary is constrained by
`competition/models/ontology.py` with a test asserting prohibited terms never
appear in machine output.

---

## 1. The question

Not "which model is most accurate", but:

> Which existing AgriVision V2 model gives the best competition-runtime
> trade-off across visual-condition performance, out-of-domain retention,
> deployability, CPU latency, artifact size, dependency burden and
> reproducibility?

The answer is not the most accurate model, and the reasoning for that is set out
below rather than glossed.

## 2. Candidate inventory

Every figure is read from a committed research artifact. No research file was
modified. Full inventory in
`competition/evaluation/results/phase2/candidate_inventory.json`.

| Candidate | In-domain | Worst external | Runtime weights | Framework deps | ONNX / `cv2.dnn` |
| --- | --- | --- | --- | --- | --- |
| **MobileNetV3-Large** | 93.0% | **70.4%** | **11.6 MB** | torch, torchvision | **yes** |
| EfficientNet-B0 | 94.6% | 68.8% | 15.6 MB | torch, torchvision | yes |
| ResNet50 | 94.2% | 69.3% | 90.0 MB | torch, torchvision | yes |
| Custom CNN (V2 baseline) | 74.1% | 34.6% | 18.4 MB | torch | yes |
| DINOv2 + linear probe | — | 66.4% | **330 MB** encoder | torch, transformers | no |
| CLIP + linear probe | — | 55.2% | **571 MB** encoder | torch, open-clip, timm | no |
| SigLIP2 + linear probe | — | 73.5% | **1431 MB** encoder | torch, transformers | no |
| SigLIP2 strict zero-shot | 97.9% | **86.3%** | **1431 MB** encoder | torch, transformers | no |
| V1 Keras CNN (2018) | — | — | 55.3 MB | TF 1.12, Py 3.6.7 | no |

In-domain figures: `v2/results/experiment_00{2,3,4}/metrics.json` and
`v2/results/baseline_test_metrics.json`. External figures: Experiment 014,
`multi_domain_summary.json`.

The decisive column is **runtime weights**. A linear probe is about 20 KB, but a
probe is useless without its encoder, and the encoder is 330 MB to 1431 MB plus
`transformers` and `torch`. The self-contained CNNs carry their whole model in
11–90 MB.

## 3. The OpenCV DNN investigation

Because this is an OpenCV competition, the first question asked of the small
CNNs was whether inference itself could run through OpenCV:

```
PyTorch checkpoint -> ONNX export -> cv2.dnn.readNetFromONNX -> inference
```

**It works, and it is numerically sound.**

| Check | Result |
| --- | --- |
| `state_dict` load into a rebuilt architecture | 0 missing, 0 unexpected |
| ONNX export (opset 17) | 11,906,184 bytes (11.35 MB) |
| `cv2.dnn.readNetFromONNX` | loads |
| `cv2.dnn` vs `onnxruntime`, same tensor | **200/200** classes agree, mean max logit diff **2.9e-06** |
| End-to-end vs torch + torchvision | **200/200** classes agree |

The consequence is the main architectural result of this phase: **the serving
image needs no deep-learning framework at all.** `torch` appears in exactly two
competition modules, both outside the serving path: `competition/models/export.py`
(build-time export) and `competition/evaluation/phase2_selection.py` (the parity
check, which imports it defensively and degrades to SKIPPED without it). Nothing
under `competition/models/adapter.py` or `competition/agent/` imports it.

The export path was not forced. It was adopted because export succeeded cleanly,
predictions are materially equivalent, and performance is competitive — all
measured below. Had any of those failed, framework-native inference was the
fallback.

## 4. Parity

Tolerances were fixed before results were examined: class agreement ≥ 0.99 and
mean worst-class probability difference ≤ 0.05.

| Comparison | Images | Class agreement | Mean max ‖Δp‖ | Worst max ‖Δp‖ |
| --- | --- | --- | --- | --- |
| Competition path vs torch + torchvision | 200 | **1.000** | 0.0129 | 0.158 |
| `cv2.dnn` vs `onnxruntime` (same tensor) | 200 | **1.000** | logit Δ 2.9e-06 | 1.9e-05 |

**Passes.** But the headline must be stated precisely, because the obvious
shorter version of it would be false.

### The preprocessing caveat

**OpenCV preprocessing is NOT equivalent to torchvision preprocessing at the
tensor level.** Mean worst-pixel divergence is **1.10** in normalised units,
because PIL antialiases when downscaling and OpenCV's `INTER_LINEAR` does not.
Typical normalised ImageNet values span roughly [-2.1, 2.6], so this is a
substantial per-pixel difference, not rounding.

Measured across interpolation modes on 40 test images (mean worst-pixel
divergence against the torchvision reference):

| Mode | Divergence |
| --- | --- |
| `INTER_AREA` | **0.648** |
| `INTER_LINEAR` | 0.686 |
| `INTER_LANCZOS4` | 0.766 |
| `INTER_CUBIC` | 0.771 |

**`INTER_AREA` is an engineering decision, not a default.** It was chosen for two
reasons: it is the antialiasing-appropriate mode for downscaling, which is what
this contract always does (research images are ~300-480 px reduced to a 256 px
shorter side), and it measured closest to the reference. It reduces divergence by
about 5% relative to `INTER_LINEAR` — a real but modest improvement. **No tested
mode achieves tensor parity.**

The supportable claim is therefore narrow:

> Prediction-level parity was preserved on the evaluated sample despite
> measurable preprocessing tensor differences.

All tested interpolation modes gave 200/200 class agreement, which suggests the
result is not delicately dependent on the mode chosen. The mean worst-class
probability difference was 0.0129, with a worst case of 0.158 — meaning at least
one image moved a class probability by ~16 points without changing the argmax.
That image was closer to flipping than the headline suggests.

**This result must not be generalised.** It holds for *this* model, on *this*
data, at *this* sample size. A model with sharper decision boundaries, a
finer-grained class set, or greater sensitivity to high-frequency detail could
convert the same tensor divergence into changed predictions. Any future model
change requires this parity measurement to be repeated, not inherited. If exact
tensor parity ever becomes necessary, the options are to reproduce PIL's
antialiased resampling in the preprocessing path or to move resizing into the
exported graph.

## 5. Runtime benchmarks

Local CPU, Python 3.11.2, Linux x86_64, OpenCV 5.0.0, 224×224 input, 100 samples
after 5 discarded warm-ups. **Not an AWS measurement.**

| Runtime | Cold load | Warm median | Warm p95 |
| --- | --- | --- | --- |
| `cv2.dnn` | **166.6 ms** | 17.52 ms | 32.18 ms |
| `onnxruntime` | 483.2 ms | **12.30 ms** | 21.19 ms |

`onnxruntime` is about 30% faster warm; `cv2.dnn` loads roughly 2.9× faster
cold. Cold load matters for the Phase 4 container decision, warm latency matters
for throughput, and both are far below any plausible budget for this workload.
`cv2.dnn` is selected — the difference is immaterial here, and executing through
OpenCV is worth more to this entry than 5 ms.

## 6. Dependency impact

| Path | Serving dependencies | Approximate weight |
| --- | --- | --- |
| **Selected: ONNX + `cv2.dnn`** | opencv-python, numpy | 11.4 MB model |
| Framework-native CNN | + torch, torchvision | ~200 MB+ |
| Foundation probe / zero-shot | + torch, transformers, encoder | 330–1431 MB model alone |

`torch`, `torchvision`, `onnx` and `onnxruntime` are installed in the
development environment for export and parity checking only, and are recorded in
`requirements-competition.txt` as build-time entries, separate from the serving
set.

## 7. Selection

**MobileNetV3-Large (V2 Experiment 004), exported to ONNX, executed via `cv2.dnn`.**

Selected **for the competition runtime** because:

1. Serving carries no deep-learning framework — the largest container lever available.
2. Smallest deployable artifact with acceptable behaviour: 11.35 MB against 1431 MB.
3. Best worst-external accuracy among self-contained CNNs (70.4%).
4. Parity with the framework reference is measured at 200/200.
5. Inference through OpenCV deepens substantive OpenCV 5 usage.

**The accepted trade-off, stated plainly:** SigLIP2 strict zero-shot holds
**86.3%** on its worst external domain against the selected model's **70.4%**.
That is a 15.9-point gap and it was knowingly given up for a 126× reduction in
model weight and the removal of `torch` and `transformers` from serving. The
selected model is *more likely to be wrong on unfamiliar produce photography* —
which is exactly why the capture gate and the escalation path exist, and why
this is recorded as a limitation rather than a footnote.

Full record, including every rejected candidate and the conditions that should
trigger a revisit: `competition/evaluation/results/phase2/selection_decision.json`.

## 8. Adapter and evidence contract

The rest of the system talks to `ConditionModel`, never to a framework class.

```python
model = load_condition_model(artifact_dir, runtime="opencv_dnn")
evidence = predict_condition(model, image_bgr)
```

`ConditionModelEvidence` carries `model_id`, `model_family`, `model_version`,
`runtime`, `input_sha256`, `predicted_research_label`, `fruit_type`,
`visible_condition`, `confidence`, `class_probabilities`,
`preprocessing_version`, `model_artifact_fingerprint`, `ontology_version` and
`inference_ms`. It is JSON-serialisable, identifies images by content hash
rather than path, and excludes timing from its deterministic payload.

A model artifact is a directory of `model.onnx` plus `manifest.json`. **The
manifest is committed; the graph is not.** The research repository excludes all
trained weights from version control, and these weights are fine-tuned on a
dataset whose licence is recorded as Unknown with no redistribution grant, so
`*.onnx` is gitignored and the export is a documented build step.

Loading fails loudly rather than degrading: a missing graph, an unreadable
graph, a fingerprint mismatch or a preprocessing mismatch against the source
checkpoint all raise.

## 9. Class ontology

Six research labels map to `fruit_type` × `visible_condition`:

| Research label | Fruit | Visible condition |
| --- | --- | --- |
| `fresh_apple` / `rotten_apple` | apple | fresh / deteriorated |
| `fresh_banana` / `rotten_banana` | banana | fresh / deteriorated |
| `fresh_orange` / `rotten_orange` | orange | fresh / deteriorated |

The research label is always reported alongside the mapped ontology, so nothing
is silently relabelled. A label outside the six raises rather than being coerced
into one of them. **The three-fruit, two-state limitation is preserved
explicitly** — no severity scale is inferred, and no safety semantics are added.

## 10. Quality-gate integration

The model does not run on every image.

```
assess capture quality  ->  remediate if permitted  ->  gate  ->  infer
                                                        |
                                                   refuse + escalate
```

Blocking flags (information destroyed): `IMAGE_TOO_SMALL`, `BLUR_RISK`,
`UNDEREXPOSED`, `OVEREXPOSED`, `SHADOW_CLIPPING`, `HIGHLIGHT_CLIPPING`.
`LOW_CONTRAST` is **advisory**: it degrades the signal without removing it, and
it is the most common residual after tone remediation. That split is a
provisional judgement call, recorded as such.

Inference runs on the canonical image — the remediated one when remediation was
accepted, the original otherwise — and which one was used is recorded on the
result and asserted by test via content hash.

## 11. Real-image evaluation

Local research images, read in place, **never copied or committed**. Outputs
contain content hashes, labels, predictions, probabilities and aggregates — no
image bytes.

| Dataset | Licence | Used | Why |
| --- | --- | --- | --- |
| Frozen V1 split (Kaggle) | **Unknown**, no redistribution grant | yes, locally | Same local-evaluation use as the research programme |
| Sultana 2022 | CC BY 4.0 | available | Permissive with attribution |
| FruitVision | **CC BY-NC-ND 4.0** | **no** | NonCommercial *and* NoDerivatives make competition-context use ambiguous, and the standing instruction is not to use ambiguous data |

396 images, balanced across the six classes, seed 42:

| Arm | Inferred | Blocked | Accuracy on inferred |
| --- | --- | --- | --- |
| Ungated (model only) | 396 | — | **94.7%** |
| Gated (capture gate active) | 12 | 384 | 91.7% |

The ungated 94.7% sits close to the research figure of 93.0% for the same model
on the full 2698-image test split, which is a useful consistency check on the
whole export, preprocessing and inference chain.

The gated arm's accuracy is computed over 12 images and is **not** a meaningful
performance number. It is reported to show the gate's effect, not the model's.

## 12. The main finding: the capture gate is badly miscalibrated for real photographs

Measured on the **validation split** — the development split; the frozen test
split is never used for calibration:

| Measure | Value |
| --- | --- |
| Images with no quality flag | **3.3%** (9 of 270) |
| Images blocked by the gate | **96.7%** |
| `HIGHLIGHT_CLIPPING` rate | 80.0% |
| `SHADOW_CLIPPING` rate | 67.8% |
| `BLUR_RISK` rate | 19.3% |

| Metric | p05 | median | p95 |
| --- | --- | --- | --- |
| `shadow_clip_fraction` | 0.000 | **0.131** | 0.283 |
| `highlight_clip_fraction` | 0.001 | **0.279** | 0.604 |
| `laplacian_variance` | 16.0 | 430.1 | 2994.1 |
| `mean_luminance` | 0.452 | 0.667 | 0.896 |

The provisional clipping limit is 0.05. The **median** real photograph clips
0.131 of shadows and 0.279 of highlights, so the gate rejects almost everything.

**The cause is diagnostic, not merely a bad constant.** These are product
photographs with bright backgrounds and dark surrounds. Whole-image clipping
statistics therefore measure *the background*, not the produce. No threshold
value fixes that: the metrics must be restricted to a foreground region to mean
anything on real captures.

That is a direct argument for the surface-segmentation work deferred to a later
phase, now supported by measurement rather than by assertion. Threshold
recalibration alone would paper over it, and calibration belongs to Phase 6 in
any case; the finding is recorded here and the thresholds are left unchanged.

## 13. Limitations

- The capture gate is unusable on real photography as currently configured
  (§12), so the end-to-end gated path is not yet demonstrable on real data.
- Preprocessing parity is not achieved at the tensor level (§4); it holds at the
  prediction level for this model on this data, which is a weaker guarantee than
  it may appear.
- The selected model is 15.9 points worse than the best candidate on the worst
  external domain (§7).
- Evaluation uses 396 images from one dataset; FruitVision is excluded and the
  Sultana external arm has not yet been run.
- No self-captured imagery exists yet, so nothing has been measured on the
  capture conditions the system is actually designed for.
- Only MobileNetV3 was exported. EfficientNet-B0 and ResNet50 are believed
  exportable and are supported by the exporter, but this is unverified.
- Peak RSS was not measured; only artifact size and latency.
- `REQUEST_HUMAN_REVIEW` is still never selected by any policy path.

## 14. Reproduction

```bash
python3 -m venv .venv-competition
.venv-competition/bin/pip install -r requirements-competition.txt

# build-time: export the research checkpoint (requires torch; not needed at serving)
.venv-competition/bin/python -m competition.models.export

# tests
.venv-competition/bin/python -m pytest tests/competition -q

# inventory, benchmarks, parity, gate calibration, condition evaluation
.venv-competition/bin/python -m competition.evaluation.phase2_selection
```

Tests that need the exported graph skip cleanly when it is absent, so a clean
clone without weights still runs the suite.

Deterministic artifacts: `candidate_inventory.json`, `parity_results.json`,
`gate_calibration.json`, `condition_eval.json`, `condition_eval.csv`,
`selection_decision.json`. Non-deterministic: `runtime_benchmarks.json` (timings).

## 15. Not done in this phase

Surface segmentation and anomaly localisation, model-versus-perception conflict
reasoning, threshold recalibration, AWS deployment, Bedrock, UI, and evaluation
on self-captured imagery.
