# Judge demo script — final

**Competition maximum: 5 minutes. This script targets 3:55.**

Measured, not guessed: **573 words of narration**, which is 3:57 at a normal
145 wpm, 4:24 at a deliberate 130 wpm, and still 4:46 at a very slow 120 wpm.
The section markers below total 3:55, leaving roughly a minute of headroom for
live uploads and pauses — the demo makes five live requests and each takes a
second or two on screen.

It is deliberately **not** written to fill the five minutes. Nothing here should
be padded to use the allowance.

Live: <https://yp2ajauzkm.us-east-1.awsapprunner.com>

Before recording, work through
[VIDEO_RECORDING_CHECKLIST.md](VIDEO_RECORDING_CHECKLIST.md) in order —
especially the warm-up run and the "no AWS console on screen" item.

**Every number spoken below is read from the screen or from the results table.
Nothing is recalled from memory, and nothing is a placeholder.**

---

## 0:00 – 0:20 · Team and problem

> *On screen: the live page, freshly loaded.*

"I'm Debalekha Chakraborty, and this is AgriVision.

Somebody holds up a phone to check whether produce has gone off. The hard part
isn't telling a good apple from a bad one — it's that most real photographs are
dim, blurred, or have six oranges in them instead of one. A classifier answers
all of those with the same confidence. It can't tell 'this looks rotten' from
'I can't see this'."

---

## 0:20 – 0:45 · What's different

> *Point at the contract line under the upload box.*

"AgriVision decides **what to do next** before it decides anything about the
fruit. OpenCV 5 measures the capture, and a deterministic policy chooses:
correct it, ask for another, escalate to a person, or go ahead.

No language model is involved. The actions are a closed list of fifteen tools.
And the scope is stated up front — one primary item per photograph."

---

## 0:45 – 1:15 · A live inspection

> *Upload the prepared image. Let it run on camera.*

"This is running on AWS now, not a recording.

Segmentation found the subject, quality was measured **on the fruit** rather
than the whole frame, nothing crossed a threshold, so the agent proceeded and
the model ran.

The classifier's answer isn't the headline — the agent's decision is."

---

## 1:15 – 2:15 · Same photograph, four actions

> *Trigger the counterfactual. This is the centre of the demo — don't rush it.*

"Now the same image, with only the pixels changed. Say it clearly: **this is a
controlled demonstration.** I'm degrading this deliberately. It isn't a sample
of how often real cameras fail.

Same subject, same policy — identical fingerprints. Four different actions:

unchanged, it proceeds. Underexposed, it applies gamma correction and
re-measures. Contrast flattened, CLAHE, and re-measures. Severely blurred — it
asks for a recapture, **and never runs the model at all.**

That last one is the point. Blur destroys the detail the assessment depends on.
Nothing to recover, so it refuses."

---

## 2:15 – 2:45 · Why it did that

> *Expand the trace on the blurred run.*

"Every decision records what caused it.

High-frequency ratio **0.255**, against a floor of **0.318** calibrated before
this photograph existed. Threshold crossed, action selected, model skipped.

And the other metrics barely moved from the reference. One measurement changed,
and it's the one that changed the outcome.

Each piece of evidence also carries how much it's trusted. Glare is advisory —
reported, never blocking on its own."

---

## 2:45 – 3:10 · OpenCV 5 and AWS

> *Switch to the architecture diagram.*

"One container on App Runner. FastAPI serving the page and the API. The
eighteen-state orchestrator. OpenCV 5 doing perception, and the classifier
running through OpenCV's own DNN module — no PyTorch in the serving image.

The model comes from S3 and its SHA-256 is verified at start-up; on a mismatch
the service reports itself not ready rather than inferring. Traces to DynamoDB,
structured logs to CloudWatch. No Bedrock, no API Gateway, and no AWS
credentials in the container."

---

## 3:10 – 3:35 · What was measured

> *Switch to the canonical results table.*

"We froze the system, then opened 38 fresh photographs it had never seen — no
overlap with earlier data on image, creator or content hash — plus 48 scenarios
whose expected actions were written down first.

Fruit type **27 of 28 where the model ran**; that denominator matters, ten
images were blocked before it. Every decision carried its evidence, **58 of
58**. Model invoked on a blocked capture: **zero of 86**. Severe blur: **12 of
12** refused.

And 10 of 12 subjects changed action when only the condition changed."

---

## 3:35 – 3:55 · Limits, and what it doesn't claim

> *On screen: the responsible-use statement.*

"Where it's weak: segmentation is the binding limit — about a quarter of real
photographs give no usable subject region. One preregistered route, the
underexposure correction, scored **2 of 12**, and we left that number alone
rather than retuning afterwards.

And it reports **visible surface condition**. It doesn't detect pathogens or
contamination, says nothing about internal spoilage, and does not determine
whether food is safe to eat. A person stays responsible for that."

---

## Things that must not be said

| Do not say | Say instead |
| --- | --- |
| "96 percent accurate" | "27 of 28 where the model ran" |
| "determines whether food is safe to eat" | "reports visible surface condition" |
| "the AI recovers the lost detail" | "it corrects the image and measures again" |
| "it detects glare" (as a blocking capability) | "glare is advisory — reported, never blocking on its own" |
| "this is how often cameras fail" | "this is a controlled demonstration" |

## If something goes wrong on camera

Keep it and narrate it, or re-record the whole take. Do not cut to imply a
result that did not happen. A live failure explained honestly is better
evidence than a clean take that hides one — and a run that requests a recapture
is the system working, not the system breaking.
