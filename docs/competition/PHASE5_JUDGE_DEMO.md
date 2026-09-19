# Phase 5 — Judge-facing demo

**Status: COMPLETE — UNCOMMITTED.** 846 tests pass (804 prior, unchanged).

**Live:** https://yp2ajauzkm.us-east-1.awsapprunner.com

No backend redesign. The orchestrator, the policies and every threshold are what
Phase 3, 3b and 4 committed. This phase adds one page and one endpoint.

---

## 1. Architecture

The page is served by the **existing** FastAPI application on the **existing**
App Runner service. No Amplify, no CloudFront, no separate SPA host, no second
container. One page and one API behind one certificate is less to deploy, less
to secure and less to explain to a judge.

Three static files — `index.html`, `app.css`, `app.js` — mounted at
`/static`, with `GET /` returning the page. No build step, no bundler, no
framework. The page is 5.4 KB and the whole payload is 30 KB.

One new endpoint: `POST /counterfactual`.

---

## 2. The page

| Section | What it shows |
| --- | --- |
| Hero | What the system does, and **one primary fruit per image** |
| Status | Live `/ready` — *Service ready · OpenCV 5.0.0 · model verified* |
| 1 · Submit | Drag-drop or picker, JPG/PNG, 12 MB, with a note on PNG being lossless |
| 2 · Agent outcome | The terminal state, and whether the model was invoked |
| 3 · Why | Evidence → interpretation → decision → next step |
| 4 · Foreground | Validity, fraction, method, and mask provenance |
| 5 · Enhancement | Only when remediation ran; before/after metrics |
| 6 · Path taken | Only the states this run entered |
| 7 · Causal trace | Per-step timeline, with raw JSON behind a disclosure |
| Counterfactual | The central demonstration |
| Technical details | Stack, bounds, fingerprints, run id |
| Responsible use | The claim boundary, unavoidable |

**The agent outcome is the headline and the classifier is not.** A produce demo
that leads with a prediction is the thing this project argues against.

---

## 3. The counterfactual

`POST /counterfactual` derives four controlled variants **in memory** from the
submitted image and runs the full bounded agent on each, through the same
orchestrator and the same frozen policy.

Measured live on the deployed service:

| Variant | First action | Model |
| --- | --- | --- |
| Reference | `NONE` | invoked |
| Underexposed | `APPLY_GAMMA` | invoked |
| Severe blur | `REQUEST_RECAPTURE` | **skipped** |
| Recoverable contrast loss | `APPLY_CLAHE` | invoked |

**Four distinct next actions from one subject under one policy.** That is the
Agentic Vision claim reduced to something a judge can watch happen.

### Two deliberate choices

**Variants come from the user's own upload.** No bundled demonstration
photograph means no licensing problem, no attribution to get wrong, and nothing
persisted. It also makes the demonstration work on a judge's own fruit.

**The contrast variant is aimed, not guessed.** The recoverable band between
the severe limit (0.12) and the calibrated advisory limit (0.1314) is narrow,
and a fixed factor overshoots it — 0.22 lands in severe territory on a typical
capture, which is what the first implementation did. The factor is now derived
from the contrast the reference run measured, with one correction step, because
`reduce_contrast` compresses toward the whole-image mean and the resulting ROI
contrast is not linear in the factor. Constructing a variant to sit at a
specific evidence level is what a *controlled* demonstration means; it is not a
claim that real captures cluster there, and the page says so.

---

## 4. Evidence maturity in the interface

Badges carry the Phase 2d/3 distinction into the UI:

| Badge | Meaning |
| --- | --- |
| **CALIBRATED** | Met the preregistered criteria; may block |
| **PROVISIONAL** | Phase 1 value retained; may block, and is marked |
| **ADVISORY** | Below its detection floor; **may not block** |
| **UNQUALIFIED** | Recorded only; may not influence any decision |

Glare renders as *Advisory* with an explicit line that it **did not block** the
result. A recovered fallback mask renders as *Provisional*, never as primary.
Asserted by test, because the honest thing to do with a detector that missed its
floor is to show it and label it, not to hide it or to promote it.

---

## 5. Failure is not an error

Human-directed outcomes are presented as instructions:

| Outcome | What the page says |
| --- | --- |
| `REQUEST_RECAPTURE` | "Please retake the photo — in focus, well lit, filling a good part of the frame." |
| `REQUEST_REPOSITION_LIGHT` | "A strong local highlight is sitting on the subject. Move the light or the camera." |
| Invalid foreground | "Could not establish a trustworthy single-subject foreground." |

No stack traces, no reason codes shown raw, and no red error styling on what is
a correct refusal. These return HTTP 200 because the inspection succeeded — the
photograph was the problem.

---

## 6. Live smoke matrix

All six against the deployed service, submitted as PNG:

| Scenario | HTTP | Outcome | Model | Trace |
| --- | ---: | --- | --- | --- |
| Normal | 200 | COMPLETE | invoked | yes |
| Underexposed | 200 | COMPLETE | invoked | yes |
| Severe blur | 200 | REQUEST_RECAPTURE | **skipped** | yes |
| Low contrast | 200 | COMPLETE | invoked | yes |
| Foreground failure | 200 | REQUEST_RECAPTURE | **skipped** | yes |
| Glare advisory | 200 | COMPLETE | invoked | yes |

**Underexposed and low contrast complete here** where the Phase 4 JPEG smoke run
sent them to recapture. That is the lossless-transport work landing: PNG
preserves the exact array, so API results match the in-process suite. The
difference was transport, never policy.

`competition/evaluation/results/phase5/`

---

## 7. Latency

| Request | Observed |
| --- | ---: |
| Page | 164 ms · 5.4 KB |
| Stylesheet | 163 ms · 9.0 KB |
| Script | 166 ms · 16.0 KB |
| Inspection (upload → result) | 298–414 ms |
| Counterfactual (four full runs) | 512 ms |

Client-observed over the public internet including TLS. One instance stays
provisioned, so there is no cold start in normal use; the page nonetheless says
a first request after idle may take longer rather than promising sub-second.

---

## 8. What the UI must never do

- **No fabricated results.** Every rendered value comes from a live response;
  a test asserts no hard-coded outcome exists in the script.
- **No internal identifiers.** A test asserts `s3://`, `arn:aws:`,
  `amazonaws.com`, `dynamodb`, the account id and filesystem paths appear in
  neither the page nor the script.
- **No food-safety claim outside the disclaimer.** The disclaimer may name what
  it denies; a test strips that section and checks the rest.
- **No full state graph on the main page.** Only the states a run entered.

---

## 9. Limitations

**Browser rendering was verified by a human, not by a test.** No headless
browser is available in this environment, so the automated evidence asserts the
served markup, the assets and the live API responses — never pixels. A person
opened the live URL and confirmed page, layout and interactions at desktop and
narrow/mobile width: `HUMAN_UI_SMOKE_CHECK = PASS`, recorded in
`demo_evidence.json`. It is a one-off manual check and does not re-run in CI, so
a future change to the page can regress rendering without any test failing.

**No screenshots captured**, and no automated visual-regression coverage.

**The endpoint is unauthenticated.** Acceptable for a judged demo. A
`DEMO_ACCESS_TOKEN` could be added through environment configuration without
harming judge access, but it is not implemented: it would be unused complexity.

**Counterfactual cost scales with variants.** Four full inspections per click,
about 512 ms. Fine for a demo, not a pattern to generalise.

**Glare demonstration depends on the upload.** The advisory path appears only if
the submitted image actually produces highlight evidence; it is not forced.

**No mobile device testing.** The layout is responsive and uses relative units,
but has not been opened on a phone.

---

## 10. Reproduction

```bash
# local
docker build -t agrivision:phase5 .
docker run -d -p 8080:8080 \
  -v "$(pwd)/competition/models/artifacts/mobilenetv3_large_v2exp004:/opt/agrivision/model:ro" \
  agrivision:phase5
open http://localhost:8080/

# tests
.venv-competition/bin/python -m pytest tests/competition/test_judge_ui.py -q
```

Demo sequence: [`JUDGE_DEMO_SCRIPT.md`](JUDGE_DEMO_SCRIPT.md).
