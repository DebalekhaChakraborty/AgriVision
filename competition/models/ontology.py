"""Competition class ontology.

The research models were trained on six classes: three fruit types crossed with
two visible condition states. That limitation is preserved explicitly rather
than papered over — the ontology here does not invent categories the model was
never trained to distinguish.

**Claim boundary.** `visible_condition` describes what is visible on the surface
of the produce in the photograph. It is not a food-safety judgement. Nothing in
this module may be read as a statement about pathogens, toxins, contamination,
internal spoilage, or whether an item is safe to eat.
"""

from __future__ import annotations

from enum import Enum

ONTOLOGY_VERSION = "phase2-ontology-1.0.0"


class FruitType(str, Enum):
    APPLE = "apple"
    BANANA = "banana"
    ORANGE = "orange"


class VisibleCondition(str, Enum):
    """Visible surface condition. Deliberately not a safety or edibility scale."""

    FRESH = "fresh"
    DETERIORATED = "deteriorated"


# Research label -> competition ontology. The research labels are read, never
# rewritten; this mapping lives on the competition side only.
RESEARCH_LABEL_TO_ONTOLOGY: dict[str, tuple[FruitType, VisibleCondition]] = {
    "fresh_apple": (FruitType.APPLE, VisibleCondition.FRESH),
    "fresh_banana": (FruitType.BANANA, VisibleCondition.FRESH),
    "fresh_orange": (FruitType.ORANGE, VisibleCondition.FRESH),
    "rotten_apple": (FruitType.APPLE, VisibleCondition.DETERIORATED),
    "rotten_banana": (FruitType.BANANA, VisibleCondition.DETERIORATED),
    "rotten_orange": (FruitType.ORANGE, VisibleCondition.DETERIORATED),
}

SUPPORTED_RESEARCH_LABELS: tuple[str, ...] = tuple(RESEARCH_LABEL_TO_ONTOLOGY)

# Human-facing wording. The research label "rotten" is mapped to
# "visible surface deterioration" for presentation; the underlying research
# label is always reported alongside it so nothing is silently relabelled.
CONDITION_DESCRIPTION: dict[VisibleCondition, str] = {
    VisibleCondition.FRESH: "no visible surface deterioration detected",
    VisibleCondition.DETERIORATED: "visible surface deterioration detected",
}

# Wording that must never appear in machine output. Enforced by test.
PROHIBITED_CLAIM_TERMS: tuple[str, ...] = (
    "safe to eat",
    "edible",
    "edibility",
    "food safety",
    "pathogen",
    "toxin",
    "contamination",
    "contaminated",
    "bacteria",
    "microbial",
    "internal spoilage",
    "shelf life",
)


class OntologyError(ValueError):
    """Raised when a label falls outside the supported ontology."""


def map_research_label(label: str) -> tuple[FruitType, VisibleCondition]:
    """Map a research class label into the competition ontology.

    Raises rather than guessing: a label the research model was not trained on
    must not be silently coerced into one that it was.
    """
    try:
        return RESEARCH_LABEL_TO_ONTOLOGY[label]
    except KeyError:
        raise OntologyError(
            f"{label!r} is outside the supported ontology; "
            f"supported: {', '.join(SUPPORTED_RESEARCH_LABELS)}"
        ) from None


def describe_condition(condition: VisibleCondition) -> str:
    return CONDITION_DESCRIPTION[condition]
