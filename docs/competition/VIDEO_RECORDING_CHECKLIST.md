# Video recording checklist

Work through this **in order** immediately before recording. The service is
live, so several of these checks can fail on the day without anything having
changed in the repository.

**The video is recorded by a person. Nothing here is automated.**

---

## 1. Before opening anything

- [ ] Confirm the competition's current maximum video length and note it here: `______`
- [ ] Target runtime **4:00**, leaving margin below that maximum
- [ ] Confirm the required link visibility (public or unlisted) and that judges
      can open it without an account

## 2. Service readiness

- [ ] `curl https://yp2ajauzkm.us-east-1.awsapprunner.com/health` → `200`, `alive`
- [ ] `curl https://yp2ajauzkm.us-east-1.awsapprunner.com/ready` → `ready: true`,
      `model_artifact_verified: true`, `persistence_healthy: true`
- [ ] App Runner service status is `RUNNING` with no failed deployment pending
- [ ] Page, `app.css` and `app.js` all return `200`

**If `/ready` is false, stop.** Recording a demo that cannot infer wastes the
take and the screen will show it.

## 3. Warm the service

- [ ] **Run one normal inspection before recording.** The first request after an
      idle period pays start-up cost and will look slow on camera. This is a
      warm-up, not a rehearsal of a fabricated result.
- [ ] Note the warm latency so the narration matches what the screen does

## 4. Demo image provenance

- [ ] Choose the demo image by the policy in [DEMO_ASSET_POLICY](#demo-asset-policy) below
- [ ] Licence verified **before** the image appears on camera
- [ ] Attribution ready, and on screen or in the description if required
- [ ] Image obeys the one-primary-item contract

## 5. Browser and desktop hygiene

- [ ] Fresh browser profile or a clean window — no bookmarks bar, no other tabs
- [ ] **No AWS console visible at any point**
- [ ] No account ID, ARN, access key, IAM page or billing page on screen
- [ ] No local filesystem paths, terminal history or editor with credentials open
- [ ] Notifications silenced
- [ ] Zoom set so the trace text is legible at the final video resolution
- [ ] Screen recorder captures the browser only, not the whole desktop

## 6. Page state

- [ ] Page loaded fresh, scrolled to the top
- [ ] Responsible-use statement reachable without hunting for it
- [ ] One-primary-fruit contract visible near the upload control
- [ ] Counterfactual control ready to trigger
- [ ] Technical trace disclosure collapsed at the start, so expanding it on
      camera shows the layering rather than a wall of JSON

## 7. Material to have open in another window

- [ ] Architecture diagram ([ARCHITECTURE.md](ARCHITECTURE.md)) rendered
- [ ] Agent workflow diagram ([AGENT_WORKFLOW_DIAGRAM.md](AGENT_WORKFLOW_DIAGRAM.md)) rendered
- [ ] Canonical results table ([TECHNICAL_REPORT.md §13](TECHNICAL_REPORT.md)) ready
- [ ] Limitations slide or section ready — this is spoken, not skipped

## 8. Content discipline while recording

- [ ] **Every number spoken is read from the screen or from the results table.**
      Do not recall a figure from memory.
- [ ] State the denominator whenever an accuracy is spoken — "27 of 28 where the
      model ran", never "96 percent accurate"
- [ ] Say **controlled demonstration** when the counterfactual variants appear
- [ ] Do not say the system determines whether food is safe to eat
- [ ] Do not describe remediation as recovering or restoring lost information
- [ ] If a live run behaves unexpectedly, **keep it and narrate it**, or re-record
      the whole take — never cut to imply a result that did not happen

## 9. After recording

- [ ] Watch the full take before uploading
- [ ] Confirm no credential, account ID or local path appears in any frame
- [ ] Confirm runtime is under the competition maximum
- [ ] Upload, set the required visibility, and open the link from a logged-out
      browser to confirm a judge can reach it
- [ ] Record the final URL in [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)

---

## Demo asset policy

In order of preference:

1. **The presenter's own photograph.** No third-party licence, no attribution
   question, and it demonstrates the real use case.
2. **CC0 or public domain.** Clean to show, no attribution obligation.
3. **Permissively licensed with clear attribution**, shown on screen or in the
   description.
4. **A controlled derivative** of one of the above, with the same attribution
   plus a modification statement.

Prefer 1 or 2. Share-alike licences create obligations on the presentation
material itself, and that complication is avoidable when a CC0 or
public-domain alternative exists.

**The evaluation corpus is never changed for presentation convenience.** Track R
and Track C are frozen. Demo asset selection is a presentation decision and has
no connection to them.
