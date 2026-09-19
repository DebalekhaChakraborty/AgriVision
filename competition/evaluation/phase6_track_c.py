"""Phase 6 Track C: controlled degradations of fresh real photographs.

Track R asks what the frozen system does with natural images. Track C asks a
narrower question that natural images cannot answer: when the visual condition
is *known*, does the agent choose the action that condition calls for?

The bases are real photographs from the frozen Track R manifest, so the subject,
the background and the camera are all real. Only the degradation is synthetic,
and it is drawn from the Phase 1 primitives that every earlier phase used - no
new transform family is invented for Phase 6.

What this is, stated plainly because the wording matters
-------------------------------------------------------
These are **controlled degradations applied to real images**. They are not real
camera-failure prevalence, not phone-camera validation, and not a sample of the
real environmental distribution. A result here says the routing is correct when
the condition is known; it says nothing about how often that condition occurs.

Preregistration
---------------
Expected actions are fixed in `phase6_expected_actions.json`, with a
fingerprint, before any variant is executed. The fingerprint exists so that a
later edit is detectable: an expectation rewritten after seeing a result is not
an expectation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from competition.service.counterfactual import _flatten_to_recoverable, _roi_contrast
from competition.vision.degradation import (
    DEGRADATION_VERSION,
    adjust_exposure,
    gaussian_blur,
)

TRACK_C_VERSION = "phase6-track-c-1.0.0"

CORPUS_ROOT = Path("competition/data/licensed_real")
VARIANT_DIR = CORPUS_ROOT / "phase6_variants"
RESULTS_DIR = Path("competition/evaluation/results/phase6")
SPLIT_MANIFEST = RESULTS_DIR / "phase6_split_manifest.json"
EXPECTED_ACTIONS = RESULTS_DIR / "phase6_expected_actions.json"

# Fixed before the bases were chosen. Selection must not be able to react to how
# any image looks or to how the system performs on it.
SELECTION_SALT = "phase6-track-c-20260919"
BASES_PER_FRUIT = 4

CLAIM_BOUNDARY = (
    "CONTROLLED DEGRADATIONS APPLIED TO REAL IMAGES. Not real camera-failure "
    "prevalence, not phone-camera validation, and not the real-world "
    "environmental distribution."
)


class TrackCError(RuntimeError):
    """Raised when the preregistration or the base selection is inconsistent."""


# --- base selection -----------------------------------------------------------


def _rank(image_id: str) -> str:
    """Stable, appearance-independent ordering key."""
    return hashlib.sha256(f"{SELECTION_SALT}:{image_id}".encode("utf-8")).hexdigest()


def select_bases(records: list[dict], per_fruit: int = BASES_PER_FRUIT) -> list[dict]:
    """Pick bases by salted hash, stratified by fruit.

    Deterministic and independent of every pixel and every metric, so the
    selection cannot drift toward images the system happens to handle well. One
    base per source group is already guaranteed by the Track R manifest.
    """
    by_fruit: dict[str, list[dict]] = {}
    for record in records:
        by_fruit.setdefault(record["fruit_type"], []).append(record)
    chosen: list[dict] = []
    for fruit in sorted(by_fruit):
        ordered = sorted(by_fruit[fruit], key=lambda r: _rank(r["image_id"]))
        chosen.extend(ordered[:per_fruit])
    if len({r["source_group_id"] for r in chosen}) != len(chosen):
        raise TrackCError("selected bases share a source group")
    return chosen


# --- the variant ladder -------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """One controlled condition and the behaviour preregistered for it."""

    key: str
    description: str
    expected_first_action: str
    condition_model_must_run: bool
    expected_terminal: str
    rationale: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "description": self.description,
            "expected_first_action": self.expected_first_action,
            "condition_model_must_run": self.condition_model_must_run,
            "expected_terminal": self.expected_terminal,
            "rationale": self.rationale,
        }


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "REFERENCE",
        "The photograph as harvested, re-encoded losslessly to PNG.",
        "NONE", True, "COMPLETE",
        "Nothing is wrong with the capture, so no corrective tool should be "
        "selected and the condition model should be reached.",
    ),
    VariantSpec(
        "UNDEREXPOSED_RECOVERABLE",
        "Exposure gain 0.12, which puts ROI mean luminance below the "
        "calibrated underexposure floor while detail survives.",
        "APPLY_GAMMA", True, "COMPLETE",
        "Underexposure is recoverable by construction, so the agent should "
        "correct it and then reach the model rather than ask for a recapture.",
    ),
    VariantSpec(
        "SEVERE_BLUR",
        "Gaussian blur sigma 6.0, which puts the ROI high-frequency ratio "
        "below the calibrated floor of 0.3182.",
        "REQUEST_RECAPTURE", False, "REQUEST_RECAPTURE",
        "Detail destroyed by defocus cannot be restored by any enhancement, so "
        "the agent must ask for a new capture and must NOT run the model.",
    ),
    VariantSpec(
        "LOW_CONTRAST_RECOVERABLE",
        "Tonal range compressed, aimed by a measurement on this image, to sit "
        "between the severe limit 0.12 and the advisory limit 0.1314.",
        "APPLY_CLAHE", True, "COMPLETE",
        "The recoverable contrast band is the band CLAHE exists for, so the "
        "agent should enhance and then reach the model.",
    ),
)

VARIANT_KEYS: tuple[str, ...] = tuple(spec.key for spec in VARIANT_SPECS)


def build_variant(key: str, image: np.ndarray) -> np.ndarray:
    """Produce one controlled variant of a base image."""
    if key == "REFERENCE":
        return image
    if key == "UNDEREXPOSED_RECOVERABLE":
        return adjust_exposure(image, 0.12)
    if key == "SEVERE_BLUR":
        return gaussian_blur(image, 6.0)
    if key == "LOW_CONTRAST_RECOVERABLE":
        # Aimed by a measurement on this image, exactly as the live demo does.
        # A fixed factor overshoots into the severe band on most photographs,
        # which would test the recapture route a second time instead of the
        # enhancement route this variant exists to exercise.
        return _flatten_to_recoverable(image, _roi_contrast(image))
    raise TrackCError(f"unknown variant {key!r}")


# --- preregistration ----------------------------------------------------------


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def preregister(bases: list[dict]) -> dict:
    """Freeze expected behaviour before a single variant is executed."""
    scenarios = [
        {"base_id": base["image_id"], "fruit_type": base["fruit_type"],
         "variant": spec.key, "expected_first_action": spec.expected_first_action,
         "condition_model_must_run": spec.condition_model_must_run,
         "expected_terminal": spec.expected_terminal}
        for base in bases
        for spec in VARIANT_SPECS
    ]
    document = {
        "track_c_version": TRACK_C_VERSION,
        "degradation_version": DEGRADATION_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "selection_salt": SELECTION_SALT,
        "bases_per_fruit": BASES_PER_FRUIT,
        "base_count": len(bases),
        "variant_count": len(VARIANT_SPECS),
        "scenario_count": len(scenarios),
        "variants": [spec.to_dict() for spec in VARIANT_SPECS],
        "scenarios": scenarios,
        "denominators": {
            "primary": (
                "All scenarios. A base whose REFERENCE never produces a valid "
                "foreground still counts, and its variants count as failures of "
                "the preregistered expectation. Perception failing before the "
                "routing question is reached is a real limitation of the system, "
                "not an excuse for excluding the case."
            ),
            "secondary": (
                "Scenarios whose base produced a valid foreground on REFERENCE. "
                "Reported alongside the primary figure, never instead of it, "
                "because it separates 'the agent routed wrongly' from 'the agent "
                "never got far enough to route'. Both denominators are fixed "
                "here, before any scenario runs."
            ),
        },
        "rule": (
            "Expected actions are frozen at this point and may not be edited "
            "after a result is seen. A scenario that does not match its "
            "expectation is recorded as a failure and classified in the failure "
            "taxonomy; it is never reconciled by rewriting the expectation."
        ),
    }
    document["expected_actions_fingerprint"] = _fingerprint(scenarios)
    return document


def write_preregistration(document: dict, bases: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_ACTIONS.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split = {
        "track_c_version": TRACK_C_VERSION,
        "selection_salt": SELECTION_SALT,
        "rule": (
            "Bases are ordered by sha256(salt:image_id) within each fruit and "
            "the first four taken. Independent of appearance and of every "
            "metric, and fixed before any variant was executed."
        ),
        "track_c_bases": [
            {"image_id": b["image_id"], "fruit_type": b["fruit_type"],
             "source_group_id": b["source_group_id"],
             "content_sha256": b["content_sha256"],
             "width": b["width"], "height": b["height"],
             "file_extension": b["file_extension"]}
            for b in bases
        ],
        "track_c_base_count": len(bases),
        "membership_note": (
            "Track C bases are also Track R images. The two tracks are reported "
            "separately and are never pooled into one headline number: Track R "
            "measures natural-domain behaviour, Track C measures routing under "
            "known conditions."
        ),
    }
    split["split_fingerprint"] = _fingerprint(split["track_c_bases"])
    SPLIT_MANIFEST.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return EXPECTED_ACTIONS, SPLIT_MANIFEST


def materialise(bases: list[dict], source_dir: Path) -> dict[tuple[str, str], Path]:
    """Write every variant to disk as PNG.

    PNG throughout: Track C depends on the variant reaching the service with the
    exact pixels it was built with, and JPEG demonstrably moves the
    high-frequency ratio the quality gate reads.
    """
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[tuple[str, str], Path] = {}
    for base in bases:
        path = source_dir / f"{base['image_id']}{base['file_extension']}"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise TrackCError(f"could not read base {base['image_id']}")
        for key in VARIANT_KEYS:
            variant = build_variant(key, image)
            destination = VARIANT_DIR / f"{base['image_id']}__{key}.png"
            if not cv2.imwrite(str(destination), variant):
                raise TrackCError(f"could not write {destination.name}")
            written[(base["image_id"], key)] = destination
    return written


def main() -> int:
    from competition.evaluation.phase6_source_pool import PHASE6_RAW, load_manifest

    manifest = load_manifest()
    bases = select_bases(manifest["records"])
    document = preregister(bases)
    expected_path, split_path = write_preregistration(document, bases)
    written = materialise(bases, PHASE6_RAW)

    print(f"bases            : {len(bases)} "
          f"({', '.join(sorted({b['fruit_type'] for b in bases}))})")
    print(f"scenarios        : {document['scenario_count']}")
    print(f"expected actions : {document['expected_actions_fingerprint']}")
    print(f"split            : {json.loads(split_path.read_text())['split_fingerprint']}")
    print(f"variants written : {len(written)} -> {VARIANT_DIR}")
    print(f"preregistered    : {expected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
