# AgriVision Agentic Vision workflow

The competition workflow diagram. One capture enters; one of six terminal states
leaves. Every branch is taken on an OpenCV measurement, and the measurement that
caused it is named on the edge.

Nothing here is decided by a language model. Actions are members of a closed
enum and tools resolve through a registry that refuses anything else, so the set
of things the system can do is fixed before it ever sees an image.

## The loop

```mermaid
flowchart TD
    A([Capture received]) --> B{validate_input<br/>decodable BGR?}
    B -- no --> FS([FAILED_SAFE<br/>no result claimed])
    B -- yes --> C[segment_foreground<br/>OpenCV subject isolation]

    C --> D{mask passes<br/>validity guards?}
    D -- "no<br/>NO_DOMINANT_COMPONENT<br/>MASK_FRAGMENTED<br/>EXCESSIVE_BORDER_CONTACT" --> IVE[INSUFFICIENT_VISUAL_EVIDENCE<br/><i>coverage failure, not occlusion</i>]
    IVE --> RC1([REQUEST_RECAPTURE<br/>or REQUEST_HUMAN_REVIEW])

    D -- yes --> E[assess_capture_quality<br/>ROI-restricted, Phase 2c-B locked policy]
    E --> F[assess_local_highlights<br/>ADVISORY]
    F --> G[assess_visibility<br/>RECORDED ONLY]
    G --> H{decide_capture_action<br/>precedence}

    H -- "IMAGE_TOO_SMALL" --> RC2([REQUEST_RECAPTURE])
    H -- "severe clipping<br/>shadow or highlight > 0.10" --> RC2
    H -- "BLUR_RISK<br/>high_frequency_ratio < 0.3182" --> RC2
    H -- "UNDEREXPOSED<br/>mean luminance < 0.0527" --> REM[apply_gamma_correction]
    H -- "residual blocking flags" --> RC2
    H -- "SEVERE_LOW_CONTRAST<br/>contrast < 0.12" --> RC2
    H -- "MODERATE_LOW_CONTRAST<br/>contrast < 0.1314" --> REM2[apply_clahe]
    H -- "nothing blocking" --> ELIG

    REM --> RS[segment_foreground<br/><b>re-perception, mandatory</b>]
    REM2 --> RS
    RS --> RA[assess_capture_quality<br/>on the new canonical image]
    RA --> RV{reassess_capture<br/>harm guard}
    RV -- "rejected<br/>revert canonical" --> RC3([REQUEST_RECAPTURE])
    RV -- "mask destroyed" --> IVE
    RV -- accepted --> H2{decide again<br/><i>no further remediation offered</i>}
    H2 -- still blocking --> RC3
    H2 -- acceptable --> ELIG

    ELIG[ELIGIBLE_FOR_INFERENCE] --> MODEL{run_condition_model<br/>MobileNetV3 via cv2.dnn}
    MODEL -- "artifact absent" --> HR([REQUEST_HUMAN_REVIEW])
    MODEL -- "raises" --> FS
    MODEL -- ok --> CI[CONDITION_INFERRED<br/>visible condition + confidence<br/><i>confidence recorded, never thresholded</i>]
    CI --> DONE([COMPLETE])

    ELIG -.->|"GLARE_RISK present"| ADV[/advisory: REQUEST_REPOSITION_LIGHT<br/>does not block/]
    ADV -.-> DONE

    style FS fill:#f8d7da,stroke:#842029
    style RC1 fill:#fff3cd,stroke:#664d03
    style RC2 fill:#fff3cd,stroke:#664d03
    style RC3 fill:#fff3cd,stroke:#664d03
    style HR fill:#fff3cd,stroke:#664d03
    style DONE fill:#d1e7dd,stroke:#0f5132
    style ADV fill:#e2e3e5,stroke:#41464b
    style IVE fill:#f8d7da,stroke:#842029
```

## What the diagram is claiming

Three things, each checkable against
[`results/phase3/`](../../competition/evaluation/results/phase3/):

**The branch depends on the pixels.** The counterfactual experiment runs one
base subject through six degradations under an identical policy. Four different
first actions come out. Nothing but the image changed.

**A refused capture cannot reach the model.** `run_condition_model` is reachable
only from `ELIGIBLE_FOR_INFERENCE`, and the orchestrator raises if called from
anywhere else. Unsafe inference rate on the scenario suite is 0/12.

**Correction forces re-perception.** There is no edge from `REASSESSED` back to
`REMEDIATION_SELECTED`, so a second automated attempt is unreachable rather than
merely discouraged. Evidence measured before a pixel changed is never read after.

## Evidence maturity on the edges

The diagram does not distinguish thick edges from thin ones, so the trace does.
Every step carries the maturity of what it measured:

| Maturity | May block? | What it is here |
| --- | --- | --- |
| `CALIBRATED` | yes | Blur floor, ROI contrast, underexposure, foreground guards, severe contrast |
| `PROVISIONAL` | yes | Overexposure and both clipping limits — calibration attempted, criteria not met, Phase 1 value retained |
| `ADVISORY` | no | Local glare evidence. 0.500 detection on unseen groups against a 0.70 floor |
| `UNQUALIFIED` | never | Subject-visibility geometry; model confidence |

A blocking decision is checked against this table when it is constructed. That
is why the glare edge in the diagram is dotted: it reaches the result, not the
gate.

## Seeing it run

The loop above is live at **https://yp2ajauzkm.us-east-1.awsapprunner.com**, and the counterfactual section of that page
runs it four times on one subject:

| Variant | Next action | Condition model |
| --- | --- | --- |
| Reference | `NONE` | invoked |
| Underexposed | `APPLY_GAMMA` | invoked after re-assessment |
| Severe blur | `REQUEST_RECAPTURE` | **skipped** |
| Recoverable contrast loss | `APPLY_CLAHE` | invoked after re-assessment |

Same subject, same policy, same fingerprints. The diagram's branches are not
illustrative — those are the edges being taken.

## Where this goes next

The loop is local and deterministic. A later phase may put a language model
*beside* it to narrate a trace for a reader — never inside it, because a
narrator that can select an action is a narrator that can select the wrong one.

Full method, results and limitations:
[`PHASE3_AGENTIC_ORCHESTRATOR.md`](PHASE3_AGENTIC_ORCHESTRATOR.md).
