"""Human-readable rendering of a machine-readable trace.

The structured trace is the artifact that matters; this turns it into something
a person can read in a terminal or a demo without a viewer. It is deliberately
a formatter and nothing more — it reads fields that are already present and adds
no interpretation, no summary judgement and no language model. If a sentence
here says a threshold was crossed, the number and the threshold are both in the
trace and both printed, so the reader can check rather than trust.

Everything that gates is printed with its evidence maturity. A reader should not
have to know which detectors are calibrated to understand which findings carried
weight, and an ADVISORY line that looks exactly like a CALIBRATED one would
undo the care taken everywhere else.
"""

from __future__ import annotations

from competition.agent.decisions import EvidenceMaturity
from competition.agent.state import InspectionState

RENDERER_VERSION = "phase3-render-1.0.0"

_MATURITY_NOTE = {
    EvidenceMaturity.CALIBRATED.value: "calibrated",
    EvidenceMaturity.PROVISIONAL.value: "provisional threshold",
    EvidenceMaturity.ADVISORY.value: "ADVISORY - cannot block",
    EvidenceMaturity.UNQUALIFIED.value: "RECORDED ONLY - cannot influence a decision",
}

# Metrics worth printing beside a step, with the label a reader understands.
_METRIC_LABELS = (
    ("high_frequency_ratio", "ROI high-frequency ratio"),
    ("focus_floor", "Threshold"),
    ("mean_luminance", "ROI mean luminance"),
    ("contrast_score", "ROI contrast"),
    ("shadow_clip_fraction", "Shadow clipping"),
    ("highlight_clip_fraction", "Highlight clipping"),
    ("foreground_fraction", "Subject fraction"),
    ("component_count", "Mask components"),
    ("largest_component_fraction", "Largest highlight"),
    ("max_local_luminance_excess", "Peak local excess (L*)"),
    ("hull_fill", "Hull fill"),
    ("solidity", "Solidity"),
    ("confidence", "Model confidence"),
    ("verdict", "Verdict"),
)


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def render_step(step, include_timing: bool = False) -> str:
    """One step as a short block of text."""
    lines = [f"Step {step.step_id} — {step.tool_name}"]
    if step.state_before != step.state_after:
        lines.append(f"  State: {step.state_before} -> {step.state_after}")

    summary = step.evidence_summary or {}
    for key, label in _METRIC_LABELS:
        if key in summary and summary[key] is not None:
            lines.append(f"  {label}: {_format_value(summary[key])}")

    for key in ("quality_flags", "artifact_flags", "invalid_reasons",
                "insufficiency_reasons", "flag_reasons"):
        if summary.get(key):
            lines.append(f"  {key.replace('_', ' ').capitalize()}: "
                         f"{_format_value(summary[key])}")

    if summary.get("glare_flag") is not None:
        lines.append(f"  Glare finding: {'yes' if summary['glare_flag'] else 'no'}")
    if summary.get("visibility_sufficient") is not None:
        lines.append(
            f"  Visibility sufficient: "
            f"{'yes' if summary['visibility_sufficient'] else 'no'}"
        )
    if summary.get("inference_executed") is not None:
        lines.append(
            f"  Condition model invoked: "
            f"{'YES' if summary['inference_executed'] else 'NO'}"
        )

    if step.evidence_maturity:
        note = _MATURITY_NOTE.get(step.evidence_maturity, step.evidence_maturity)
        lines.append(f"  Evidence maturity: {step.evidence_maturity} ({note})")
    if step.selected_action:
        lines.append(f"  Action: {step.selected_action}")
    if step.decision_reason:
        lines.append(f"  Reason: {step.decision_reason}")
    if include_timing and step.duration_ms:
        lines.append(f"  Duration: {step.duration_ms:.1f} ms")
    return "\n".join(lines)


def render_trace(trace, include_timing: bool = False) -> str:
    """The whole trace as readable text."""
    blocks = [render_step(step, include_timing) for step in trace.steps]
    return "\n\n".join(blocks)


def render_result(result, include_timing: bool = False) -> str:
    """Trace plus the headline facts a reader wants first."""
    header = [
        "=" * 70,
        f"INSPECTION {result.run_id}",
        "=" * 70,
        f"Final state       : {result.final_state.value}",
        f"Condition model   : {'INVOKED' if result.inference_ran else 'NOT INVOKED'}",
    ]
    if result.foreground_invalid_reasons:
        header.append(
            f"Foreground        : INVALID "
            f"({', '.join(result.foreground_invalid_reasons)})"
        )
    else:
        header.append(f"Foreground        : {'valid' if result.foreground_valid else 'not assessed'}")
    if result.quality_flags:
        header.append(f"Quality flags     : {', '.join(result.quality_flags)}")
    if result.artifact_flags:
        header.append(f"Artefact findings : {', '.join(result.artifact_flags)}")
    if result.remediation_applied:
        header.append(
            f"Remediation       : {result.remediation_action} "
            f"({'accepted' if result.remediation_accepted else 'REJECTED'})"
        )
    if result.requested_human_action:
        header.append(f"Human action      : {result.requested_human_action} (required)")
    if result.advisory_human_action:
        header.append(
            f"Advisory          : {result.advisory_human_action} "
            "(suggestion only; the result was not blocked on it)"
        )
    if result.condition_evidence:
        header.append(
            f"Visible condition : "
            f"{result.condition_evidence.get('visible_condition')} "
            f"({result.condition_evidence.get('fruit_type')}), "
            f"confidence {result.model_confidence:.3f} — recorded, not thresholded"
        )
    if result.failure:
        header.append(f"Failure           : {result.failure}")
    header.append("-" * 70)
    body = render_trace(result.trace, include_timing)
    return "\n".join(header) + "\n\n" + body + "\n"


def render_summary_line(result) -> str:
    """One line, for a scenario table."""
    return (
        f"{result.final_state.value:26s} "
        f"model={'Y' if result.inference_ran else 'N'} "
        f"human={result.requested_human_action or '-':24s} "
        f"steps={len(result.trace.steps)}"
    )
