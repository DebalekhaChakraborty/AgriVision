# Judge demo sequence

**Live:** https://yp2ajauzkm.us-east-1.awsapprunner.com

A demonstrable sequence, not a spoken script. Roughly 3–4 minutes, so it fits
inside a 5-minute submission video with room for titles and a closing card.

Everything below is performed against the live service. Nothing is precomputed.

---

## 0:00–0:20 · The problem

Open the page. The hero states it:

> OpenCV 5 evaluates whether the visual evidence is trustworthy, takes bounded
> corrective actions when appropriate, and runs the condition model **only**
> when the capture is suitable.

**The point to land:** a classifier that answers confidently on a photograph it
should have refused is the failure this project exists to avoid. Most produce
inspection demos skip straight to a prediction.

Show the input contract directly beneath: **one primary fruit per image.**

---

## 0:20–0:40 · What the system is

Point at the service badge in the header — *Service ready · OpenCV 5.0.0 · model
verified* — which is a live `/ready` call, not a graphic.

Say what the loop is: perception, then a deterministic decision, then a tool.
**No language model participates in any decision.** One remediation attempt, one
model invocation, eighteen named states.

---

## 0:40–1:20 · A normal inspection

Drop in a clear photograph of one fruit. Press **Inspect**.

Walk the result top to bottom:

1. **Agent outcome** — *Inspection complete*, and beside it *Condition model
   invoked*. The agent outcome is the headline; the classifier is not.
2. **Why** — the evidence-to-action chain: the measured high-frequency ratio
   against the calibrated floor, the interpretation, the decision, the next step.
3. **Visible-condition model** — fruit, visible condition, confidence, with the
   note that confidence is *recorded, not thresholded*.

If a judge asks why confidence isn't used to escalate: no confidence policy has
been calibrated, so acting on it would be inventing a threshold.

---

## 1:20–2:30 · The counterfactual — the heart of the demo

Press **Show Agentic Counterfactual** on the *same* image.

The server derives four controlled variants in memory and runs the full bounded
agent on each. Read the grid across:

| Variant | OpenCV finding | Next action | Model |
| --- | --- | --- | --- |
| Reference | no blocking finding | `NONE` | invoked |
| Underexposed | ROI luminance below the calibrated floor | `APPLY_GAMMA` | invoked after re-assessment |
| Severe blur | high-frequency ratio below 0.3182 | `REQUEST_RECAPTURE` | **skipped** |
| Recoverable contrast loss | ROI contrast in the recoverable band | `APPLY_CLAHE` | invoked after re-assessment |

**The sentence that matters:** *same subject, same policy, same fingerprints —
four different next actions, because the visual evidence changed.*

Point at the amber notice above the grid. It says these are controlled
variants, not a benchmark. Say that out loud; do not let it pass as fine print.

---

## 2:30–3:10 · The causal trace

Scroll to **Causal trace** on the blurred variant, or re-inspect a blurred
capture.

Step through: segment foreground → assess ROI quality → decide → *condition
model skipped*. Each step carries its tool, its numbers and a maturity badge.

Expand **Technical trace** to show the raw JSON a judge can verify, then point
out what is *not* in it: no image bytes, no filenames, no filesystem paths. The
image is identified by content hash.

Then show the maturity badges and explain the distinction in one line:
**CALIBRATED** evidence may block, **ADVISORY** evidence may not. Glare is
advisory because it reached 0.500 detection on unseen groups against a
preregistered floor of 0.70 — it is reported, never gating.

---

## 3:10–3:40 · Technical proof

Expand **Technical details**: OpenCV 5.0.0, MobileNetV3-Large through
`cv2.dnn`, AWS App Runner, a deterministic bounded state machine, one
remediation attempt, one model invocation, and the policy fingerprints for this
run.

Worth saying: OpenCV 5 is verified *inside* the container across seventeen
operations, and the model is fetched from S3 and SHA-256 verified at startup —
a mismatch leaves the service unready rather than running an unverified graph.

---

## 3:40–4:00 · Limitations and responsible use

Scroll to the responsible-use panel and read it:

> AgriVision assesses **visible produce condition**. It does not detect
> pathogens, toxins, microbiological contamination or internal spoilage, and it
> does not determine whether food is safe to eat.

Then state the honest limits, because they are the credible part:

- **Segmentation is the binding constraint.** On independent validation only 12
  of 29 photographs produced a mask that passed its guards. The system refuses
  rather than guessing.
- **One fruit per image.** Market stalls and piles are out of scope, measured:
  pushing the fallback past that boundary produced masks adjudicated wrong on
  18 of 19 scenes.
- **Glare and subject-visibility detectors did not meet their floors** and are
  not permitted to gate.

---

## Fallback if the network fails

`competition/evaluation/results/phase3/traces/` holds six curated traces in
JSON and readable text. Label them **recorded example** on screen; do not let a
stored trace look like a live run.

---

## What not to say

- Not "detects spoilage" — **visible condition**.
- Not "AI restored the lost detail" — **capture enhancement attempted, evidence
  re-assessed**.
- Not "99% accurate" — the scenario suite is a deterministic behaviour result,
  not accuracy.
- Not "fully solved" — segmentation is the open problem, and saying so is more
  convincing than claiming otherwise.
