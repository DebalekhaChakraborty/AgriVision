# Phase 3 — Bounded Agentic Vision orchestrator

**Status: COMPLETE — UNCOMMITTED.** 730 tests pass (591 prior, unchanged).

Phase 1 measured captures. Phase 1b decided what to do about them. Phase 2b
restricted the measurement to the subject, Phase 2c-B calibrated the thresholds
on licensed real photographs, and Phase 2d added local artefact evidence. Each
produced a number or a rule.

This phase produces **behaviour**: an explicit loop in which an OpenCV
measurement determines which tool runs next, and in which the things the system
must not do are unreachable rather than merely unwritten.

---

## 1. What is new, and what is reused

Nothing in `competition/vision/` was rewritten. The orchestrator calls the
existing modules; the Phase 1b tone policy still decides exposure, and the
Phase 2c-B locked policy still supplies every ROI threshold.

Four things are new:

| Module | What it adds |
| --- | --- |
| `competition/agent/state.py` | The state machine as a value, with a declared transition table |
| `competition/agent/decisions.py` | Evidence maturity, and the decision contract that checks it |
| `competition/agent/tools/registry.py` | The closed tool vocabulary with contracts and failure modes |
| `competition/agent/orchestrator.py` | The loop, the budget, and the fail-safe boundary |
| `competition/agent/render.py` | Structured trace → readable text, with no model involved |

One small addition was made to a Phase 1 module: `validate_image` and
`image_content_sha256` in `competition/vision/quality.py`, public wrappers over
functions the module already used. The agent needs to check and identify an
image *without* measuring it, and delegating means its notion of "valid" cannot
drift from the evidence layer's.

---

## 2. The state machine

Previously the state of an inspection was implicit — it lived in whichever
combination of local variables happened to be set inside `inspect_capture`.
That is adequate for a fixed pipeline and inadequate for an agent, because
nothing can be asserted about a state that is never named. "The classifier did
not run on a blocked capture" was true by construction rather than by check.

```
RECEIVED → INPUT_VALIDATED → FOREGROUND_ASSESSED → QUALITY_ASSESSED
        → ARTIFACTS_ASSESSED → DECISION_MADE
              ├→ REMEDIATION_SELECTED → REMEDIATED → RESEGMENTED → REASSESSED ┐
              └──────────────────────────────────────────────────────────────┴→
        → ELIGIBLE_FOR_INFERENCE → CONDITION_INFERRED → COMPLETE
```

Terminals: `COMPLETE`, `REQUEST_RECAPTURE`, `REQUEST_REPOSITION_LIGHT`,
`REQUEST_HUMAN_REVIEW`, `FAILED_SAFE`.

**`INSUFFICIENT_VISUAL_EVIDENCE` is deliberately not terminal.** It records what
the system knows — that no measurement it trusts could be taken — and says
nothing about what anyone should do. The policy then routes it onward to a state
that does name an action. Keeping the two separate means the *reason* a capture
was refused stays legible independently of the remedy proposed for it.

Three properties follow from the table rather than from discipline:

- A second automated remediation is **unreachable**: `REASSESSED` has no edge to
  `REMEDIATION_SELECTED`.
- Re-perception after a correction is **mandatory**: the only way out of
  `REMEDIATED` is `RESEGMENTED`.
- Inference is reachable **only** from `ELIGIBLE_FOR_INFERENCE`, and `COMPLETE`
  only from `CONDITION_INFERRED`.

---

## 3. The tool registry

Fifteen tools, each declaring its contracts, its failure states, whether it
changes the canonical image and whether it ends in a request to a person.

`resolve_tool` accepts a `ToolName` member or its exact string value and refuses
everything else — including the member *name* rather than its value, and a value
with a trailing space. This is the guarantee that survives contact with a
language model later: a narrator may describe a trace, and no sentence it
produces can name a tool into existence.

A test walks the registry and imports every declared implementation. It found a
real defect on first run — `FINALIZE_RESULT` named a function that did not exist
under that name — which is exactly the drift the test is for.

---

## 4. Evidence maturity

Two Phase 2d detectors did not earn the right to stop an inspection. Both still
produce evidence worth recording. The danger is not that they exist; it is that
a reader of a trace six months from now cannot tell which numbers were
calibrated and which were guesses, and treats them alike.

So maturity is carried in the type system. Maturities come from the Phase 2c-B
locked-policy record rather than being asserted: the four thresholds that met
the preregistered criteria are `CALIBRATED`, the three retained at their Phase 1
values are `PROVISIONAL`.

| Maturity | May gate | Evidence |
| --- | --- | --- |
| `CALIBRATED` | yes | `roi.high_frequency_ratio`, `roi.contrast_score`, `roi.mean_luminance.underexposed`, `foreground.*`, `artifact.severe_contrast`, `artifact.moderate_contrast` |
| `PROVISIONAL` | yes | `roi.mean_luminance.overexposed` (AUC 0.943 vs 0.95 floor), `roi.shadow_clip_fraction`, `roi.highlight_clip_fraction` |
| `ADVISORY` | **no** | `artifact.glare` — 0.655 controlled, 0.500 validation, 0.167 FP, floor 0.70 |
| `UNQUALIFIED` | **never** | `artifact.visibility` (0.40, never fired), `model.confidence` (no threshold calibrated) |

`AgentDecision.__post_init__` checks a blocking decision against the maturity of
the evidence that triggered it. Blocking on `ADVISORY` raises; naming
`UNQUALIFIED` evidence as a trigger raises regardless of whether the decision
blocks. **An unregistered evidence id is `UNQUALIFIED`**, so a new measurement is
powerless until someone registers it — failing in the safe direction.

### The explicit override

`ArtifactPolicy(glare_blocks=True)` is an operator deciding to accept a
0.500-detection signal. That decision is legal, but it may not be silent:
the orchestrator sets `advisory_gating_permitted=True` on the decision, which
appears in the serialised record and in every trace it touches. The override
never admits `UNQUALIFIED` evidence — visibility stays powerless however the
policy is configured.

This distinction emerged from a failing test. The guard initially refused the
policy flag outright and the run failed safe, which was safe but illegible: a
deliberate configuration looked like a crash.

---

## 5. Policy precedence

Ordered from *information destroyed* to *information degraded*. An unrecoverable
condition is never masked by a recoverable one, so a severely clipped image is
never sent for gamma correction that would redistribute what survived.

| # | Condition | Action |
| --- | --- | --- |
| 1 | Not a decodable BGR image | `FAILED_SAFE` |
| 2 | Foreground fails its validity guards | `INSUFFICIENT_VISUAL_EVIDENCE` → recapture / review |
| 3 | `IMAGE_TOO_SMALL` | `REQUEST_RECAPTURE` |
| 4 | Shadow or highlight clipping past 0.10 | `REQUEST_RECAPTURE` |
| 5 | `BLUR_RISK` with adequate photometry | `REQUEST_RECAPTURE` |
| 6 | Underexposure above the unrecoverable rung | `APPLY_GAMMA` |
| 7 | Residual blocking flags after correction | `REQUEST_RECAPTURE` |
| 8 | `SEVERE_LOW_CONTRAST` (< 0.12) | `REQUEST_RECAPTURE` |
| 9 | `MODERATE_LOW_CONTRAST` (< 0.1314) | `APPLY_CLAHE` |
| 10 | `GLARE_RISK` | advisory only, unless explicitly enabled |
| 11 | Nothing blocking | `RUN_CONDITION_MODEL` |

Step 7 restores a Phase 1b behaviour that delegation had quietly dropped. The
tone tier recaptures only on *severe* clipping; Phase 1b additionally refused
inference on any blocking flag, severe or not. A test asserts
`RESIDUAL_BLOCKING_FLAGS == DEFAULT_BLOCKING_FLAGS`, so the two layers cannot
drift apart.

---

## 6. Bounded execution

| Limit | Value | What it closes |
| --- | --- | --- |
| `max_remediation_attempts` | 1 | An agent that enhances until it likes the result |
| `max_condition_model_calls` | 1 | A second opinion that is really the same opinion |
| `max_segmentation_calls` | 2 | Initial, plus one after remediation |
| `max_steps` | 24 | Any unforeseen cycle becomes a safe stop, not a hang |

Exceeding any limit raises internally and is converted to `FAILED_SAFE` with the
breach recorded in `failure`. A test runs the loop with `max_steps=2` and
confirms it stops rather than continuing outside its declared bounds.

---

## 7. Human control

Three terminal states name an action a person must take:
`REQUEST_RECAPTURE`, `REQUEST_REPOSITION_LIGHT`, `REQUEST_HUMAN_REVIEW`.

The run **stops** there. It does not wait, poll, or simulate the person having
complied; a later capture is a new run with a new id. There is no code path that
fabricates a satisfied request.

`REQUEST_REPOSITION_LIGHT` exists because a specular highlight is not an
exposure problem and no tone curve recovers the surface beneath it. The correct
response is to change the lighting geometry, which only a person can do.

---

## 8. The counterfactual experiment

The Agentic Vision claim reduced to something falsifiable. One base subject,
one policy, identical fingerprints on every row; only the pixels differ.

| Variant | First action | Final state | Model ran | Advisory |
| --- | --- | --- | --- | --- |
| `REFERENCE` | `NONE` | COMPLETE | yes | — |
| `UNDEREXPOSED` | `APPLY_GAMMA` | COMPLETE | yes | — |
| `SEVERE_BLUR` | `REQUEST_RECAPTURE` | REQUEST_RECAPTURE | **no** | — |
| `LOW_CONTRAST` | `APPLY_CLAHE` | COMPLETE | yes | — |
| `SEVERE_CONTRAST` | `REQUEST_RECAPTURE` | REQUEST_RECAPTURE | **no** | — |
| `GLARE` | `NONE` | COMPLETE | yes | `REQUEST_REPOSITION_LIGHT` |
| `SEGMENTATION_FAILURE` | `REQUEST_RECAPTURE` | REQUEST_RECAPTURE | **no** | — |

Four distinct first actions from one photograph. `GLARE` is the informative
row: it takes the *same* action as `REFERENCE` and differs only in the advisory
it carries, which is precisely what a non-gating detector should do.

`competition/evaluation/results/phase3/counterfactual_actions.json`

---

## 9. Scenario suite and agent metrics

Twelve scenarios, each declaring its expected terminal state, model invocation,
required tools and **forbidden** tools before it runs.

| Metric | Value | Denominator |
| --- | --- | --- |
| `TASK_SUCCESS_RATE` | **1.0** | 12 scenarios |
| `DECISION_ATTRIBUTION_RATE` | **1.0** | 33 action-selecting trace steps |
| `UNSAFE_INFERENCE_RATE` | **0.0** | 12 runs |
| `UNNECESSARY_TOOL_CALL_RATE` | **0.0** | 12 runs |
| `BOUNDED_EXECUTION_RATE` | **1.0** | 12 runs |
| `TRACE_COMPLETENESS_RATE` | **1.0** | 12 traces |
| `FAIL_SAFE_RATE` | **1.0** | 5 known-unusable inputs |

> These are **deterministic scenario-suite results** on constructed fixtures with
> known expected behaviour. A task-success rate of 1.0 means the agent did what
> the policy says it should on twelve cases someone wrote down in advance. It is
> **not accuracy**, and it says nothing about how often real photographs are
> glared or blurred.

Fixtures are synthetic by design, not by convenience. The Phase 2c-B held-out
groups are spent and the Phase 2d validation set was opened once; using either
again for a new confirmatory claim would be the leakage those splits exist to
prevent. Synthetic fixtures also let a scenario state its expected behaviour
without reference to any measurement, which is what makes "expected" mean
anything.

### The one honest compromise

`rejected_remediation` uses a **policy variant**, and says so in its own record.
The harm guard never fired on any natural fixture — gamma computed to reach a
target band reaches it — so rather than contrive an image until it failed, the
acceptance guard is tightened beyond any achievable correction so the *routing*
of a rejection can be observed. It exercises orchestration, and it is not
evidence about images.

---

## 10. Latency

Local CPU, 384×384 fixtures, n=15 after one untimed warm-up pass (the ONNX graph
is lazily initialised on first inference, and letting that land inside the sample
would make the tail describe process startup).

| Path | Median | p95 |
| --- | ---: | ---: |
| `NO_REMEDIATION` | 62.9 ms | 73.0 ms |
| `REMEDIATION` | 109.0 ms | 113.8 ms |
| `RECAPTURE_TERMINAL` | 51.1 ms | 56.5 ms |
| `SEGMENTATION_FAILURE` | 9.6 ms | 11.1 ms |

Stage medians: highlights 18.6 ms, quality 12.6 ms, **model inference 10.1 ms**,
foreground 7.0 ms, visibility 4.4 ms, gamma 3.7 ms.

**Orchestration overhead is 0.7–2.2 ms** — the agent layer costs almost nothing
over the vision work it coordinates. The first measurement of this was wrong:
summing per-tool medians undercounts a path that calls each tool twice, which
every remediation path does, and it reported 48.8 ms. Overhead is now computed
per run against that run's own steps.

> **Local CPU only. This is not an AWS figure** and must never be reported as
> one: no deployment runtime has been built, sized or measured.

---

## 11. Traces

Six curated traces in `results/phase3/traces/`, each as `.json` (machine) and
`.txt` (human): normal, underexposure remediation, severe blur recapture,
segmentation failure, low-contrast remediation, glare advisory.

No filesystem path, no image bytes, no restricted imagery — images are named by
content hash only, asserted by test against `/home/`, `/Users/`, `/tmp/`, `C:\`,
`.jpg` and `.png`. Timing is excluded from the deterministic payload, so two
runs of the same input compare byte-identical while remaining measurable.

The renderer is a formatter and nothing else. It adds no interpretation and no
summary judgement, and **no language model participates**. Where a line says a
threshold was crossed, the number and the threshold are both printed, so the
reader can check rather than trust.

---

## 12. Limitations

**Fixtures are synthetic.** Every number in §9 describes constructed images. The
suite demonstrates that the loop behaves as specified; it does not show how
often real captures fall into each branch.

**Segmentation remains the binding constraint.** Phase 2d found only 12 of 29
independent validation photographs produced a valid foreground. Phase 3 does not
improve that — it makes the failure *safe* by routing it to a person instead of
measuring a region it cannot attribute. The synthetic fixtures segment far more
reliably than real photographs do, so the scenario suite does not exercise this
failure at its true rate.

**Glare remains advisory and unimproved.** 0.500 detection on unseen groups.
Phase 3 gives it a home in the trace, not authority.

**The visibility detector is still not a detector.** It runs, it is recorded,
and it is structurally prevented from influencing anything.

**Model confidence is recorded and unused.** The Phase 2c-B median of 0.416
across six classes is a domain-shift signal, not a decision rule. No
confidence-driven escalation exists, and the maturity registry prevents one
being added without calibration.

**The moderate-contrast band is narrow.** Between the severe limit (0.12) and
the calibrated advisory limit (0.1314) lies a band few real images occupy, so
the CLAHE route fires rarely.

**One scenario uses a policy variant** (§9).

**No AWS, no Bedrock, no UI.** None of this has been deployed or measured
anywhere but a local CPU.

---

## 13. Claim boundary

Supported by the artifacts in `results/phase3/`:

> OpenCV 5 perception outputs deterministically alter subsequent tool calls and
> human actions in a bounded perception–decision–action loop.

**Not** claimed:

- That Agentic Vision special-prize requirements are complete — judge-visible UI,
  AWS runtime and final evaluation all remain.
- Any real-world accuracy figure for the agent, for glare, or for visibility.
- That segmentation, glare or visibility improved in this phase. None did.
- Any food-safety, edibility or contamination judgement. The model reports
  **visible condition** only, and a test asserts the prohibited vocabulary is
  absent from every result.

---

## 14. Reproduction

```bash
# the whole evaluation
python -m competition.evaluation.phase3_agent_scenarios all

# individually
python -m competition.evaluation.phase3_agent_scenarios scenarios
python -m competition.evaluation.phase3_agent_scenarios counterfactual
python -m competition.evaluation.phase3_agent_scenarios metrics
python -m competition.evaluation.phase3_agent_scenarios traces
python -m competition.evaluation.phase3_agent_scenarios latency

# tests
python -m pytest tests/competition/ -q
```

The ONNX artifact is not committed. Without it the suite still runs: scenarios
needing inference are reported as **skipped with the reason stated**, never as
passing. Verified by hiding the artifact — 636 passed, 94 skipped, 0 failed.

Diagram: [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md).
