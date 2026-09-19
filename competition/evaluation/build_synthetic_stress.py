"""Tier 3: synthetic stress scenes, reported separately and never mixed in.

What these are
--------------
Procedurally composed scenes drawn with OpenCV primitives - shaded ellipses on
generated backgrounds, with clutter, occluders, distractors and unusual
lighting. They are **not** produced by an image-generation model; no generative
model was used anywhere in this phase. `ProvenanceType.SYNTHETIC_GENERATED`
covers both cases and the schema refuses to let either carry a real-image
licence or enter a real-image track, which is the property that matters.

What they are for
-----------------
Probing failure modes the licensed corpus happens not to contain: a subject that
touches three borders, several subjects of equal size, a distractor the same
colour as the subject, a background busier than the foreground. Classical
segmentation has characteristic ways of breaking, and a corpus of competent
photographs will not exercise them.

What they are not for
---------------------
Any statement about real-world accuracy. These scenes contain no fruit. They
cannot support a condition result, a quality threshold, or a claim about
deployment behaviour, and results from them are written to a separate file so
they cannot be quietly averaged into the real-image numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.data.licensed_sources import (
    SYNTHETIC_LICENCE_ID,
    CorpusTrack,
    LicensedImageRecord,
    ProvenanceType,
    Split,
)
from competition.evaluation.build_licensed_corpus import CORPUS_ROOT, MANIFEST_DIR

SYNTHETIC_DIR = CORPUS_ROOT / "synthetic"
SYNTHETIC_MANIFEST = MANIFEST_DIR / "synthetic_stress.json"
SYNTHETIC_RESULTS = Path("competition/evaluation/results/phase2c_synthetic_stress.json")
SYNTHETIC_VERSION = "phase2c-synthetic-stress-1.0.0"
GENERATOR = "procedural_opencv_composition"
DEFAULT_SEED = 20260918
SIZE = (768, 1024)  # height, width

# Plausible produce hues in BGR, used only so the scenes are not grey discs.
# They are not claimed to match any real fruit's colour.
SUBJECT_COLOURS = {
    "apple_like": (40, 40, 190),
    "banana_like": (60, 200, 225),
    "orange_like": (40, 140, 240),
}


@dataclass(frozen=True)
class StressCase:
    name: str
    probes: str


STRESS_CASES: tuple[StressCase, ...] = (
    StressCase("plain_background", "baseline: one subject, flat background"),
    StressCase("busy_background", "background with more texture energy than the subject"),
    StressCase("subject_touching_borders", "subject clipped by three frame edges"),
    StressCase("three_equal_subjects", "no dominant component"),
    StressCase("same_colour_distractor", "a second object the subject's colour"),
    StressCase("tiny_subject", "subject far below the minimum foreground fraction"),
    StressCase("subject_fills_frame", "subject above the maximum foreground fraction"),
    StressCase("heavy_occlusion", "an opaque bar across the subject"),
    StressCase("low_contrast_scene", "subject and background of near-equal lightness"),
    StressCase("directional_hotspot", "a blown specular patch on the subject"),
    StressCase("fragmented_subject", "subject broken into disconnected pieces"),
    StressCase("gradient_background", "smooth background ramp across the frame"),
)


def _rng(seed: int, case: str) -> np.random.Generator:
    salt = int(hashlib.sha256(f"{seed}:{case}".encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(salt)


def _background(kind: str, rng: np.random.Generator) -> np.ndarray:
    height, width = SIZE
    if kind == "busy":
        canvas = rng.integers(40, 215, (height, width, 3), dtype=np.uint8)
        return cv2.GaussianBlur(canvas, (0, 0), 1.1)
    if kind == "gradient":
        ramp = np.linspace(30, 220, width, dtype=np.float32)
        return np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2).astype(np.uint8)
    if kind == "clutter":
        canvas = np.full((height, width, 3), 150, dtype=np.uint8)
        for _ in range(28):
            x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
            w, h = int(rng.integers(40, 220)), int(rng.integers(40, 220))
            colour = tuple(int(v) for v in rng.integers(40, 220, 3))
            cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), colour, -1)
        return cv2.GaussianBlur(canvas, (0, 0), 0.8)
    return np.full((height, width, 3), int(rng.integers(120, 190)), dtype=np.uint8)


def _draw_subject(
    canvas: np.ndarray, centre: tuple[int, int], axes: tuple[int, int],
    colour: tuple[int, int, int], rng: np.random.Generator,
) -> None:
    """A shaded, lightly textured ellipse - enough structure to segment and focus on."""
    overlay = np.zeros_like(canvas)
    cv2.ellipse(overlay, centre, axes, 0, 0, 360, colour, -1)

    mask = overlay.any(axis=2)
    speckle = rng.integers(-28, 28, canvas.shape, dtype=np.int16)
    shaded = np.clip(overlay.astype(np.int16) + speckle, 0, 255).astype(np.uint8)

    # Simple lambertian-ish falloff so the subject is not a flat disc.
    yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    falloff = 1.0 - 0.45 * np.sqrt(
        ((xx - centre[0]) / max(axes[0], 1)) ** 2 + ((yy - centre[1]) / max(axes[1], 1)) ** 2
    )
    falloff = np.clip(falloff, 0.35, 1.0)[:, :, None]
    shaded = np.clip(shaded.astype(np.float32) * falloff, 0, 255).astype(np.uint8)
    canvas[mask] = shaded[mask]


def compose(case: StressCase, seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = _rng(seed, case.name)
    height, width = SIZE
    colour = SUBJECT_COLOURS[list(SUBJECT_COLOURS)[int(rng.integers(0, 3))]]

    background_kind = {
        "busy_background": "busy", "gradient_background": "gradient",
        "same_colour_distractor": "clutter", "three_equal_subjects": "clutter",
    }.get(case.name, "flat")
    canvas = _background(background_kind, rng)
    centre = (width // 2, height // 2)

    if case.name == "subject_touching_borders":
        _draw_subject(canvas, (width // 2, height // 2), (width // 2 + 60, height // 2 + 40),
                      colour, rng)
    elif case.name == "three_equal_subjects":
        for x in (width // 4, width // 2, 3 * width // 4):
            _draw_subject(canvas, (x, height // 2), (110, 110), colour, rng)
    elif case.name == "same_colour_distractor":
        _draw_subject(canvas, (width // 3, height // 2), (150, 150), colour, rng)
        _draw_subject(canvas, (2 * width // 3, height // 3), (140, 140), colour, rng)
    elif case.name == "tiny_subject":
        _draw_subject(canvas, centre, (26, 26), colour, rng)
    elif case.name == "subject_fills_frame":
        _draw_subject(canvas, centre, (width, height), colour, rng)
    elif case.name == "fragmented_subject":
        for offset in range(-3, 4):
            _draw_subject(canvas, (centre[0] + offset * 120, centre[1] + offset * 55),
                          (45, 45), colour, rng)
    elif case.name == "low_contrast_scene":
        canvas = np.full((height, width, 3), 150, dtype=np.uint8)
        _draw_subject(canvas, centre, (190, 170), (150, 152, 158), rng)
    else:
        _draw_subject(canvas, centre, (210, 180), colour, rng)

    if case.name == "heavy_occlusion":
        cv2.rectangle(canvas, (0, height // 2 - 60), (width, height // 2 + 60), (20, 20, 20), -1)
    if case.name == "directional_hotspot":
        glare = np.zeros((height, width), dtype=np.float32)
        cv2.circle(glare, (centre[0] + 70, centre[1] - 50), 110, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 45)
        canvas = np.clip(canvas.astype(np.float32) + glare[:, :, None] * 255.0,
                         0, 255).astype(np.uint8)
    return canvas


def build(seed: int = DEFAULT_SEED) -> dict:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for case in STRESS_CASES:
        image = compose(case, seed)
        digest = hashlib.sha256(
            np.ascontiguousarray(image).tobytes() + str(image.shape).encode("ascii")
        ).hexdigest()
        image_id = "SYN-" + hashlib.sha256(
            f"{seed}:{case.name}".encode("utf-8")
        ).hexdigest()[:12]
        cv2.imwrite(str(SYNTHETIC_DIR / f"{image_id}.png"), image)

        record = LicensedImageRecord(
            image_id=image_id,
            source_group_id=f"SG-SYNTHETIC-{case.name[:20]}",
            capture_family_id=f"CF-SYN-{case.name[:16]}",
            fruit_type="apple",  # nominal only; these scenes contain no fruit
            provenance=ProvenanceType.SYNTHETIC_GENERATED.value,
            track=CorpusTrack.SYNTHETIC_STRESS.value,
            source_name="AgriVision synthetic stress set",
            source_url="",
            creator=GENERATOR,
            licence_id=SYNTHETIC_LICENCE_ID,
            licence_url="",
            attribution_text="",
            commercial_use_allowed=True,
            derivatives_allowed=True,
            attribution_required=False,
            share_alike_required=False,
            retrieved_at=date.today().isoformat(),
            content_sha256=digest,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            file_extension=".png",
            local_only=True,
            split=Split.UNASSIGNED.value,
            notes=f"stress case: {case.probes}",
        )
        record.validate()
        records.append({**record.to_dict(), "case": case.name, "probes": case.probes})

    return {
        "synthetic_version": SYNTHETIC_VERSION,
        "generator": GENERATOR,
        "generator_note": (
            "drawn with OpenCV primitives; no image-generation model was used, "
            "and these scenes contain no fruit"
        ),
        "seed": seed,
        "claim_boundary": (
            "exploratory stress probes only; they support no statement about "
            "real-world accuracy, thresholds or condition inference"
        ),
        "excluded_from": ["threshold calibration", "held-out evaluation", "any reported accuracy"],
        "count": len(records),
        "records": records,
    }


def evaluate(document: dict) -> dict:
    """Run segmentation and the locked gate over the stress set, separately."""
    from competition.evaluation.calibrate_roi_policy import _measure_one, gate_sample
    from competition.evaluation.evaluate_locked_policy import gate_outcome, load_locked_policy

    policy = load_locked_policy()
    outcomes: list[dict] = []
    for entry in document["records"]:
        path = SYNTHETIC_DIR / f"{entry['image_id']}{entry['file_extension']}"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        measured = _measure_one(image)
        sample = {
            "sample_id": entry["image_id"], "kind": "base", "transformation": "none",
            "level": 1.0, "fruit_type": entry["fruit_type"], "track": entry["track"],
            "split": "synthetic", "width": entry["width"], "height": entry["height"],
            "expected_verdict": "UNLABELLED", **measured,
        }
        outcomes.append({
            "case": entry["case"],
            "probes": entry["probes"],
            "segmentation_valid": bool(measured.get("segmentation_valid")),
            "invalid_reasons": measured.get("invalid_reasons", []),
            "foreground_fraction": measured.get("foreground_fraction"),
            "component_count": measured.get("component_count"),
            "border_contact_fraction": measured.get("border_contact_fraction"),
            "roi_available": bool(measured.get("roi_available")),
            "gate_flags": gate_sample(sample, policy, "roi")["flags"],
            "gate_outcome": gate_outcome(sample, policy)["outcome"],
        })

    return {
        "synthetic_version": SYNTHETIC_VERSION,
        "policy_threshold_fingerprint": policy.threshold_fingerprint(),
        "status": "EXPLORATORY",
        "claim_boundary": document["claim_boundary"],
        "n": len(outcomes),
        "segmentation_valid": sum(1 for o in outcomes if o["segmentation_valid"]),
        "cases": outcomes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.build_synthetic_stress",
        description="Build and evaluate the synthetic stress tier (exploratory only).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)

    document = build(args.seed)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {"generated": document["count"]}

    if args.evaluate:
        results = evaluate(document)
        SYNTHETIC_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        SYNTHETIC_RESULTS.write_text(
            json.dumps(results, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        summary["segmentation_valid"] = f"{results['segmentation_valid']}/{results['n']}"

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
