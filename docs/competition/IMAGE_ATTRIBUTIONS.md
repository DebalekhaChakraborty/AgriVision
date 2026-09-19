# Image attributions — public-facing material

Attribution for every third-party image **shown publicly**: in the README, the
technical report, the demo page, or the submission video.

---

## Current position: nothing to attribute yet

**No third-party photograph is displayed in any public-facing material.**
Verified rather than assumed:

| Surface | Third-party images shown | Why |
| --- | --- | --- |
| Live demo page | **none** | the page bundles no demo imagery; every variant is derived in memory from the image the viewer uploads and nothing is persisted |
| README (competition section) | **none** | text, tables and links only |
| Technical report | **none** | text, tables and diagrams only |
| Committed figures | **none** | every committed image is a generated plot — matplotlib output from measurements, with no photographic content |
| Submission video | **to be decided at recording** | see below |

The demo page's counterfactual is the reason this stays clean. Because the
variants are generated from the viewer's own upload rather than from a bundled
sample, there is no demonstration photograph to licence, attribute or
redistribute.

---

## Where the full provenance record lives

This file covers **presentation** only. The complete, per-image provenance of
the evaluation corpus is elsewhere and is not duplicated here:

| Record | Contents |
| --- | --- |
| `competition/evaluation/results/phase6/phase6_licence_manifest.json` | all 38 Phase 6 evaluation images: creator, source URL, licence, licence URL, attribution text, commercial-use and derivative permissions, share-alike obligation, content SHA-256, track membership |
| `competition/data/licensed_real/manifests/` | the Phase 2c-B and Phase 2d corpora on the same terms |

**None of those 38 images is shown publicly**, so none is listed here. Dumping
all 38 into a presentation attribution list would imply they appear somewhere
they do not.

For reference, the Phase 6 corpus breaks down as: CC-BY-SA-4.0 ×19,
CC-BY-SA-3.0 ×6, CC-BY-2.0 ×5, CC0-1.0 ×2, public domain ×2, CC-BY-SA-2.0 ×2,
CC-BY-SA-2.5 ×1, CC-BY-4.0 ×1. All permit commercial use and derivatives; 28
carry a share-alike obligation and 34 require attribution. No Unknown, no
NonCommercial, no NoDerivatives — the licence gate rejects all three by default.

---

## If an image is used in the video

Add a row here **before** recording, not after.

| Image | Creator | Source | Licence | Licence URL | Modified? | Attribution text to display |
| --- | --- | --- | --- | --- | --- | --- |
| _(none yet)_ | | | | | | |

### Selection order

1. **The presenter's own photograph** — no third-party licence, no attribution
   question, and it demonstrates the actual use case.
2. **CC0 or public domain** — clean to show, no obligation.
3. **Permissively licensed with attribution** — display the attribution on
   screen or in the video description.
4. **A controlled derivative** of any of the above — same attribution plus an
   explicit modification statement.

Prefer 1 or 2. A share-alike licence can place obligations on the presentation
material itself, and that complication is avoidable whenever a CC0 or
public-domain alternative exists.

### Attribution format

For a CC-BY or CC-BY-SA image, displayed legibly on screen or in the
description:

```
<Creator>, "<Title>", <licence id>, via <source>
<source URL>
Modified: <what was changed, or "not modified">
```

### Rules that do not bend

- **Licence verified before the image appears on camera**, not afterwards.
- **A modification statement is required** whenever the shown image is derived —
  cropped, degraded, corrected, or recoloured. The counterfactual demonstration
  produces derivatives by design, so if a licensed image is used for it the
  derivative status must be stated.
- **The evaluation corpus is never changed for presentation convenience.** Track
  R and Track C are frozen. Choosing a demo image is a presentation decision and
  has no connection to them.
