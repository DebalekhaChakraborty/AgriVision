"""Schema for the self-captured deployment-domain image set.

Defines what a capture *is* before any capture exists, so that collection cannot
drift into whatever happened to get photographed. Nothing here calibrates a
threshold; Phase 2c-A locks the protocol, and calibration happens once real
images exist.

Two rules drive the design:

**Splits are assigned at the physical item level, never the image level.** Every
photograph of the same fruit belongs to exactly one split. A dark photo of
apple APL-003 in evaluation and a reference photo of the same apple in
calibration would leak the subject across the boundary and make held-out results
meaningless.

**Quality targets come from the capture protocol, not from the metrics.** The
label records what the photographer deliberately did — "I defocused this badly"
— so calibration compares metrics against an independent ground truth rather
than against itself.

**Claim boundary.** `VisibleConditionAnnotation` describes appearance only. It
carries no food-safety, edibility or contamination meaning, and the vocabulary
deliberately excludes such terms.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from competition.models.ontology import FruitType

CAPTURE_SCHEMA_VERSION = "phase2c-capture-schema-1.0.0"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# APL-001, BAN-012, ORG-007
ITEM_ID_PATTERN = re.compile(r"^(APL|BAN|ORG)-\d{3}$")
ITEM_PREFIX_BY_FRUIT: dict[FruitType, str] = {
    FruitType.APPLE: "APL",
    FruitType.BANANA: "BAN",
    FruitType.ORANGE: "ORG",
}
FRUIT_BY_ITEM_PREFIX = {prefix: fruit for fruit, prefix in ITEM_PREFIX_BY_FRUIT.items()}


class CaptureSchemaError(ValueError):
    """Raised when a capture record or set violates the protocol."""


class Split(str, Enum):
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class VisibleConditionAnnotation(str, Enum):
    """Photographer's conservative note on visible appearance.

    Deliberately excludes SAFE, UNSAFE, EDIBLE and CONTAMINATED. This is an
    appearance note, not a safety judgement, and `UNCERTAIN` exists so a
    photographer is never forced into a confident label.
    """

    FRESH_APPEARING = "FRESH_APPEARING"
    VISIBLY_DEGRADED = "VISIBLY_DEGRADED"
    UNCERTAIN = "UNCERTAIN"


class QualityTarget(str, Enum):
    """Preregistered capture-quality label, assigned by protocol and human review.

    Never derived from the OpenCV metrics being calibrated — that would make the
    calibration circular.
    """

    ACCEPTABLE = "ACCEPTABLE"
    REMEDIABLE = "REMEDIABLE"
    RECAPTURE_REQUIRED = "RECAPTURE_REQUIRED"
    ADVISORY_ONLY = "ADVISORY_ONLY"


class CaptureCondition(str, Enum):
    """Locked capture-condition taxonomy.

    Derived from the eleven conditions in DEMO_DATA_PLAN.md, with three
    documented changes recorded in the Phase 2c protocol note:

    * "Visibly deteriorated produce" moves from a *condition* to an *item*
      attribute (`VisibleConditionAnnotation`) — it describes the fruit, not how
      it was photographed.
    * "Mixed fruit in frame" and "Non-produce object" become scene cases rather
      than per-item conditions, since neither is a photograph of one item.
    * Blur and darkness are split into mild and severe, because the boundary
      between "remediable" and "recapture required" is exactly what calibration
      has to locate.
    """

    REFERENCE = "REFERENCE"
    REFERENCE_REPEAT = "REFERENCE_REPEAT"
    DIM = "DIM"
    DARK_SEVERE = "DARK_SEVERE"
    OVEREXPOSED = "OVEREXPOSED"
    GLARE = "GLARE"
    DEFOCUS_MILD = "DEFOCUS_MILD"
    DEFOCUS_SEVERE = "DEFOCUS_SEVERE"
    MOTION_BLUR = "MOTION_BLUR"
    CLUTTERED_BACKGROUND = "CLUTTERED_BACKGROUND"
    PARTIAL_OCCLUSION = "PARTIAL_OCCLUSION"
    SMALL_SUBJECT = "SMALL_SUBJECT"


class SceneCase(str, Enum):
    """Scene-level cases that are not photographs of a single catalogued item."""

    MIXED_FRUIT = "MIXED_FRUIT"
    NON_PRODUCE = "NON_PRODUCE"


#: Replicates required for each scene case.
SCENE_REPLICATES: int = 3
#: Total scene captures across all cases.
SCENE_CAPTURE_COUNT: int = SCENE_REPLICATES * len(SceneCase)


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the photographer and the evaluator need for one condition."""

    condition: CaptureCondition
    purpose: str
    procedure: str
    quality_target: QualityTarget
    remediation_expected: bool
    recapture_preferred: bool
    tests: str  # "quality gate", "segmentation", or "both"
    required_for_every_item: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["condition"] = self.condition.value
        data["quality_target"] = self.quality_target.value
        return data


# The locked matrix. `required_for_every_item` conditions are captured for all
# items; the rest are captured on a documented subset, to keep the collection
# burden realistic for a solo participant without losing coverage.
CONDITION_MATRIX: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        CaptureCondition.REFERENCE,
        purpose="Baseline the gate should accept; anchors every other capture of the item.",
        procedure="Ordinary indoor diffuse light, whole fruit visible, normal phone distance (~25-35 cm), plain everyday surface, no deliberate degradation.",
        quality_target=QualityTarget.ACCEPTABLE,
        remediation_expected=False,
        recapture_preferred=False,
        tests="both",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.REFERENCE_REPEAT,
        purpose="Estimate capture-to-capture repeatability of the metrics.",
        procedure="Repeat REFERENCE after lowering and re-raising the camera. Do not attempt to reproduce the framing exactly.",
        quality_target=QualityTarget.ACCEPTABLE,
        remediation_expected=False,
        recapture_preferred=False,
        tests="quality gate",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.DIM,
        purpose="Underexposure that one bounded gamma remediation should recover.",
        procedure="REDUCED ambient illumination: switch off the main room light or step away from the window. The fruit must remain CLEARLY VISIBLE to you through the viewfinder. Do not use flash, and do not push it toward darkness - that is DARK_SEVERE.",
        quality_target=QualityTarget.REMEDIABLE,
        remediation_expected=True,
        recapture_preferred=False,
        tests="quality gate",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.OVEREXPOSED,
        purpose="Highlight clipping on the subject, distinct from a bright background.",
        procedure="Raise overall exposure: strong DIFFUSE illumination on the fruit, or increase camera exposure compensation until highlights visibly blow out. Keep the light broad - do NOT create a concentrated specular hotspot, which is GLARE.",
        quality_target=QualityTarget.REMEDIABLE,
        remediation_expected=True,
        recapture_preferred=False,
        tests="quality gate",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.DEFOCUS_SEVERE,
        purpose="Blur that no tone operation can recover; must reach recapture.",
        procedure="Focus deliberately AWAY from the subject, enough to destroy inspection detail: tap-focus on something across the room, then photograph the fruit without refocusing. Surface texture should be unreadable.",
        quality_target=QualityTarget.RECAPTURE_REQUIRED,
        remediation_expected=False,
        recapture_preferred=True,
        tests="quality gate",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.CLUTTERED_BACKGROUND,
        purpose="Foreground isolation against a realistic non-plain background.",
        procedure="REFERENCE lighting and distance, with everyday objects around and behind the fruit. The SUBJECT itself must remain reasonably captured - well lit and in focus. The difficulty comes from the surrounding scene, not from the fruit.",
        quality_target=QualityTarget.ADVISORY_ONLY,
        remediation_expected=False,
        recapture_preferred=False,
        tests="segmentation",
        required_for_every_item=True,
    ),
    ConditionSpec(
        CaptureCondition.DARK_SEVERE,
        purpose="Shadow crush that is genuinely unrecoverable.",
        procedure="NEAR-DARK capture, intended to materially remove visual information and trigger recapture. Curtains closed and lights off, no flash. The fruit should be barely discernible. This is deliberately more extreme than DIM.",
        quality_target=QualityTarget.RECAPTURE_REQUIRED,
        remediation_expected=False,
        recapture_preferred=True,
        tests="quality gate",
        required_for_every_item=False,
    ),
    ConditionSpec(
        CaptureCondition.GLARE,
        purpose="Specular highlight on the fruit surface itself.",
        procedure="DIRECTIONAL light: a torch or bare bulb held at an angle specifically to produce a specular highlight - a small bright spot reflecting off the skin. Overall exposure stays normal; the difficulty is the hotspot, not general brightness.",
        quality_target=QualityTarget.ADVISORY_ONLY,
        remediation_expected=False,
        recapture_preferred=False,
        tests="both",
        required_for_every_item=False,
    ),
    ConditionSpec(
        CaptureCondition.DEFOCUS_MILD,
        purpose="The boundary case calibration must locate between acceptable and recapture.",
        procedure="SLIGHT focus error: focus just in front of or behind the fruit. The fruit must remain clearly RECOGNISABLE - marginally soft, not obviously blurred. This is the boundary case, so err toward too little blur rather than too much.",
        quality_target=QualityTarget.REMEDIABLE,
        remediation_expected=False,
        recapture_preferred=False,
        tests="quality gate",
        required_for_every_item=False,
    ),
    ConditionSpec(
        CaptureCondition.MOTION_BLUR,
        purpose="Handheld motion; not remediable by tone adjustment.",
        procedure="Move the CAMERA steadily sideways as the shutter fires, keeping the fruit nominally in frame. Directional smearing should be visible. Do not move the fruit, and do not let it leave the frame.",
        quality_target=QualityTarget.RECAPTURE_REQUIRED,
        remediation_expected=False,
        recapture_preferred=True,
        tests="quality gate",
        required_for_every_item=False,
    ),
    ConditionSpec(
        CaptureCondition.PARTIAL_OCCLUSION,
        purpose="Part of the subject hidden; segmentation and gate robustness.",
        procedure="Cover approximately one quarter to one third of the fruit with a hand or object, otherwise REFERENCE conditions. Do not move, rotate or alter the fruit itself - only occlude it.",
        quality_target=QualityTarget.ADVISORY_ONLY,
        remediation_expected=False,
        recapture_preferred=False,
        tests="both",
        required_for_every_item=False,
    ),
    ConditionSpec(
        CaptureCondition.SMALL_SUBJECT,
        purpose="Subject small or off-centre; exercises the foreground area guards.",
        procedure="Increase CAMERA DISTANCE to roughly one metre so the fruit occupies a small part of the frame, or place it near a frame corner. Do not crop or digitally resize afterwards - the small subject must come from the physical capture.",
        quality_target=QualityTarget.ADVISORY_ONLY,
        remediation_expected=False,
        recapture_preferred=False,
        tests="segmentation",
        required_for_every_item=False,
    ),
)

CONDITION_SPEC_BY_ID: dict[CaptureCondition, ConditionSpec] = {
    spec.condition: spec for spec in CONDITION_MATRIX
}
REQUIRED_CONDITIONS: tuple[CaptureCondition, ...] = tuple(
    spec.condition for spec in CONDITION_MATRIX if spec.required_for_every_item
)
SAMPLED_CONDITIONS: tuple[CaptureCondition, ...] = tuple(
    spec.condition for spec in CONDITION_MATRIX if not spec.required_for_every_item
)


# --------------------------------------------------------------------------
# Collection targets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionTarget:
    """How many physical items to obtain, and how they are allocated.

    A pragmatic solo-participant target, not a statistical power calculation.
    """

    label: str
    items_per_fruit: int
    calibration_per_fruit: int
    evaluation_per_fruit: int
    sampled_items_per_fruit: int
    note: str

    @property
    def total_items(self) -> int:
        return self.items_per_fruit * len(FruitType)

    @property
    def calibration_items(self) -> int:
        return self.calibration_per_fruit * len(FruitType)

    @property
    def evaluation_items(self) -> int:
        return self.evaluation_per_fruit * len(FruitType)

    @property
    def extended_condition_items(self) -> int:
        """How many items receive the sampled (extended) conditions."""
        return self.sampled_items_per_fruit * len(FruitType)

    @property
    def required_capture_count(self) -> int:
        return self.total_items * len(REQUIRED_CONDITIONS)

    @property
    def extended_capture_count(self) -> int:
        return self.extended_condition_items * len(SAMPLED_CONDITIONS)

    @property
    def total_capture_count(self) -> int:
        """Exact expected number of photographs, scene captures included."""
        return (
            self.required_capture_count
            + self.extended_capture_count
            + SCENE_CAPTURE_COUNT
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "total_items": self.total_items,
            "calibration_items": self.calibration_items,
            "evaluation_items": self.evaluation_items,
            "extended_condition_items": self.extended_condition_items,
            "required_capture_count": self.required_capture_count,
            "extended_capture_count": self.extended_capture_count,
            "scene_capture_count": SCENE_CAPTURE_COUNT,
            "total_capture_count": self.total_capture_count,
        })
        return data


TARGET = CollectionTarget(
    label="TARGET",
    items_per_fruit=8,
    calibration_per_fruit=5,
    evaluation_per_fruit=3,
    sampled_items_per_fruit=2,
    note=(
        "Preferred size: 24 items, 15 calibration and 9 held-out. "
        "186 photographs: 144 required + 36 extended + 6 scene."
    ),
)

MINIMUM = CollectionTarget(
    label="MINIMUM",
    items_per_fruit=6,
    calibration_per_fruit=4,
    evaluation_per_fruit=2,
    sampled_items_per_fruit=2,
    note=(
        "Acceptable fallback: 18 items, 12 calibration and 6 held-out. With only "
        "two held-out items per fruit, a single atypical fruit moves the result "
        "materially, so any evaluation figure must be reported with that caveat "
        "and no threshold should be chosen on a narrow margin."
    ),
)

STRETCH = CollectionTarget(
    label="STRETCH",
    items_per_fruit=10,
    calibration_per_fruit=6,
    evaluation_per_fruit=4,
    sampled_items_per_fruit=3,
    note="Only if items are readily available; not required.",
)

COLLECTION_TARGETS: dict[str, CollectionTarget] = {
    target.label: target for target in (TARGET, MINIMUM, STRETCH)
}


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def parse_item_id(item_id: str) -> FruitType:
    """Validate an item id and return the fruit type it encodes."""
    if not isinstance(item_id, str) or not ITEM_ID_PATTERN.match(item_id):
        raise CaptureSchemaError(
            f"invalid item id {item_id!r}; expected APL-001, BAN-012 or ORG-007 form"
        )
    return FRUIT_BY_ITEM_PREFIX[item_id.split("-")[0]]


@dataclass(frozen=True)
class CaptureRecord:
    """One photograph in the capture set.

    Deliberately carries **no filesystem path**. The manifest that drives local
    execution keeps paths in a separate LOCAL-ONLY file; anything committed or
    reported is addressed by content hash.
    """

    image_id: str
    item_id: str
    fruit_type: FruitType
    capture_condition: CaptureCondition
    replicate_id: int
    split: Split
    device_id: str
    quality_target: QualityTarget
    content_sha256: str
    width: int
    height: int
    orientation: str = "as_shot"
    visible_condition: VisibleConditionAnnotation = VisibleConditionAnnotation.UNCERTAIN
    remediation_expected: bool = False
    capture_notes: str = ""
    schema_version: str = CAPTURE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fruit_type"] = self.fruit_type.value
        data["capture_condition"] = self.capture_condition.value
        data["split"] = self.split.value
        data["quality_target"] = self.quality_target.value
        data["visible_condition"] = self.visible_condition.value
        return data

    def validate(self) -> None:
        """Raise on anything structurally wrong with this record."""
        encoded_fruit = parse_item_id(self.item_id)
        if encoded_fruit is not self.fruit_type:
            raise CaptureSchemaError(
                f"{self.item_id} encodes {encoded_fruit.value} but the record says "
                f"{self.fruit_type.value}"
            )
        if not isinstance(self.capture_condition, CaptureCondition):
            raise CaptureSchemaError(f"unknown capture condition {self.capture_condition!r}")
        if not isinstance(self.split, Split):
            raise CaptureSchemaError(f"unknown split {self.split!r}")
        if not isinstance(self.quality_target, QualityTarget):
            raise CaptureSchemaError(f"unknown quality target {self.quality_target!r}")
        if len(self.content_sha256) != 64:
            raise CaptureSchemaError("content_sha256 must be a 64-character hex digest")
        if self.width <= 0 or self.height <= 0:
            raise CaptureSchemaError(f"implausible dimensions {self.width}x{self.height}")
        if self.replicate_id < 1:
            raise CaptureSchemaError("replicate_id starts at 1")
        if not self.device_id:
            raise CaptureSchemaError("device_id is required")

        expected = CONDITION_SPEC_BY_ID[self.capture_condition]
        if self.quality_target is not expected.quality_target:
            raise CaptureSchemaError(
                f"{self.capture_condition.value} is preregistered as "
                f"{expected.quality_target.value}, not {self.quality_target.value}; "
                "change the protocol deliberately rather than per-record"
            )


@dataclass(frozen=True)
class SceneRecord:
    """A scene case that is not a photograph of one catalogued item."""

    image_id: str
    scene_case: SceneCase
    split: Split
    device_id: str
    content_sha256: str
    width: int
    height: int
    capture_notes: str = ""
    schema_version: str = CAPTURE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scene_case"] = self.scene_case.value
        data["split"] = self.split.value
        return data


def expected_item_ids(target: CollectionTarget) -> list[str]:
    """The full catalogue of item ids implied by a collection target."""
    ids: list[str] = []
    for fruit, prefix in ITEM_PREFIX_BY_FRUIT.items():
        ids.extend(f"{prefix}-{index:03d}" for index in range(1, target.items_per_fruit + 1))
    return sorted(ids)


def protocol_summary() -> dict:
    """Machine-readable snapshot of the locked protocol."""
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "status": "PROTOCOL LOCKED - CAPTURE PENDING",
        "supported_fruit": [fruit.value for fruit in FruitType],
        "targets": {name: target.to_dict() for name, target in COLLECTION_TARGETS.items()},
        "conditions": [spec.to_dict() for spec in CONDITION_MATRIX],
        "required_conditions": [c.value for c in REQUIRED_CONDITIONS],
        "sampled_conditions": [c.value for c in SAMPLED_CONDITIONS],
        "scene_cases": [case.value for case in SceneCase],
        "quality_targets": [target.value for target in QualityTarget],
        "visible_condition_vocabulary": [v.value for v in VisibleConditionAnnotation],
        "split_rule": "Splits are assigned per physical item; all captures of an item share a split.",
    }
