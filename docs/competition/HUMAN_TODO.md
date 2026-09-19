# Human-only remaining actions

Engineering is frozen. Everything below needs a person — a camera, a browser
session, or a judgement call. Nothing here can be automated, and nothing already
finished is repeated.

Supporting material is ready for every item; the links point at it.

---

## Before recording

- [ ] **Re-check the video length limit against the current official rules.**
      This package assumes **5 minutes maximum**. If that has changed, the
      script needs re-timing, not trimming on the fly.
- [ ] **Choose the demo image.** Order of preference in
      [IMAGE_ATTRIBUTIONS.md](IMAGE_ATTRIBUTIONS.md): your own photograph first,
      then CC0 or public domain. It must show **one primary fruit**.
- [ ] **If the image is third-party, add its row to IMAGE_ATTRIBUTIONS.md
      before recording**, not after. The table is empty and waiting.

## Recording

- [ ] **Work through [VIDEO_RECORDING_CHECKLIST.md](VIDEO_RECORDING_CHECKLIST.md)
      in order.** The two that actually bite: run one warm-up inspection before
      recording, and keep the AWS console off screen entirely.
- [ ] **Record the video** following
      [JUDGE_DEMO_SCRIPT.md](JUDGE_DEMO_SCRIPT.md). 573 words of narration,
      ~3:55 with the live pauses. Do not pad it to fill five minutes.
- [ ] **Watch the whole take before uploading.** Check no account ID, access
      key, local path or console appears in any frame.
- [ ] Confirm the runtime is under the limit.

## Publishing the video

- [ ] **Upload as public or unlisted** — judges must be able to open it.
- [ ] **Open the link in a logged-out browser** and confirm it plays. An
      unlisted link that silently requires sign-in fails the requirement.
- [ ] Record the URL in [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) §7.

## Submission form

- [ ] **Complete the Devpost submission.**
- [ ] Add the **repository link**, on branch `competition/opencv-aws-2026`,
      tag `opencv-aws-2026-submission`.
- [ ] Add the **live endpoint**: <https://yp2ajauzkm.us-east-1.awsapprunner.com>
- [ ] Add the **video link**.
- [ ] Attach or link the **technical report**
      ([TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)) in whatever form the form
      requires.
- [ ] **Select the Agentic Vision award track** as well as the overall entry.
- [ ] **Preview the submission** and read it as a judge would.
- [ ] **Check nothing sensitive is visible** in the preview — no credentials, no
      account identifiers beyond ARNs, no local paths.
- [ ] **Submit before the deadline.**
- [ ] **Save the confirmation** — screenshot or email.

## After submitting

- [ ] **Leave the AWS infrastructure running** until judging is over. The
      endpoint is part of the submission. Cost is roughly $0.72/month outside
      judging traffic, so there is no pressure to tear it down early.
- [ ] Keep the tag where it is. Moving or recreating it after submission breaks
      the link between what was judged and what is in the repository.

---

## Judgement calls that are yours, not mine

Two things I have deliberately left open rather than decided:

1. **Whether to publish the model artifact.** It is currently private, so a
   third party can run the 770 artifact-independent tests but cannot run
   model-dependent inference locally. The live endpoint demonstrates the full
   system, so this is a completeness question, not a functional gap. Publishing
   would strengthen the reproduction claim and means redistributing a build
   product derived from a research checkpoint.
2. **Whether to mention the ablation result in the video.** The retrospective
   ablation found the quality gate did **not** improve fruit-type classification
   on the fresh sample — the baseline got all ten images the agent declined. It
   is stated plainly in the technical report (§14). The script does not cover it,
   because in four minutes the honest framing needs more room than it has. That
   is a presentation decision, and I would rather you make it knowingly than
   discover it in the report afterwards.
