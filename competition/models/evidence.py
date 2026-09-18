"""Typed evidence produced by the condition model.

Mirrors the discipline of `competition.vision.evidence`: JSON-serialisable,
deterministic apart from timing, identified by content hash rather than path,
and carrying enough provenance that a stored prediction can be re-derived.

**Claim boundary.** `visible_condition` describes the produce surface as
photographed. It is not a safety, edibility or spoilage judgement, and the
wording used here is constrained by `competition.models.ontology`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from competition.models.ontology import (
    ONTOLOGY_VERSION,
    FruitType,
    VisibleCondition,
    describe_condition,
)

CONDITION_EVIDENCE_VERSION = "phase2-condition-evidence-1.0.0"


@dataclass(frozen=True)
class ConditionModelEvidence:
    """A single condition prediction with full provenance."""

    model_id: str
    model_family: str
    model_version: str
    runtime: str
    input_sha256: str
    predicted_research_label: str
    fruit_type: FruitType
    visible_condition: VisibleCondition
    confidence: float
    class_probabilities: dict[str, float]
    preprocessing_version: str
    model_artifact_fingerprint: str
    ontology_version: str = ONTOLOGY_VERSION
    evidence_version: str = CONDITION_EVIDENCE_VERSION
    inference_ms: float = field(default=0.0)

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        data["fruit_type"] = self.fruit_type.value
        data["visible_condition"] = self.visible_condition.value
        data["condition_description"] = describe_condition(self.visible_condition)
        if not include_timing:
            data.pop("inference_ms", None)
        return data

    def to_json(self, include_timing: bool = True, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_timing=include_timing), indent=indent, sort_keys=True
        )

    def deterministic_payload(self) -> dict:
        """Everything that must be identical across runs of the same input."""
        return self.to_dict(include_timing=False)


class ConditionModelError(RuntimeError):
    """Raised when a condition model cannot be loaded or used."""
