"""Controlled degradation of licensed real images, with preregistered labels.

Why degrade at all
------------------
Real photographs from Commons are mostly competent photographs. They contain
very few genuinely unusable captures, which is exactly the population a quality
gate needs in order to be calibrated: without any bad images, every threshold
that accepts everything scores perfectly. Controlled degradation supplies the
negative class with a known cause and a known severity.

What this buys, and what it does not
------------------------------------
Each derived sample is a *real* photograph with one *known* transformation
applied. That is much stronger evidence than a synthetic scene, and much weaker
evidence than a real bad photograph. A Gaussian blur is not a defocused lens: it
has no bokeh structure, no aberration, and it is uniform across a scene that
actually had depth. Multiplying by a gain is not underexposure: a real sensor
adds noise as it loses signal, and this transform does not. Every number
produced from this set is therefore a statement about the gate's response to a
*characterised perturbation*, not about its response to real camera failure.
The optional phone-capture validation exists to close precisely that gap.

Labels are fixed before measurement
-----------------------------------
`EXPECTED_VERDICT` below assigns ACCEPTABLE or UNUSABLE to each rung of each
ladder, written from the severity of the transformation alone. They are not
derived from any metric, and the rungs where a defensible person would disagree
are labelled BORDERLINE and excluded from the confusion matrix rather than being
forced to one side to make the numbers tidier. A rubric that is adjusted after
seeing the metrics is not a rubric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from competition.data.licensed_sources import (
    CorpusError,
    LicensedImageRecord,
    Split,
    derive_from,
)
from competition.evaluation.build_licensed_corpus import (
    DERIVED_DIR,
    MANIFEST_DIR,
    load_image,
)
from competition.evaluation.split_licensed_corpus import split_records
from competition.vision.degradation import DegradationSpec, apply_degradation

DERIVED_MANIFEST = MANIFEST_DIR / "derived_manifest.json"
DERIVATION_VERSION = "phase2c-controlled-degradation-1.0.0"
DEFAULT_SEED = 20260918


class ExpectedVerdict(str, Enum):
    """What the capture gate *should* say about a sample. Preregistered."""

    ACCEPTABLE = "ACCEPTABLE"
    UNUSABLE = "UNUSABLE"
    # Genuinely arguable. Reported, never scored.
    BORDERLINE = "BORDERLINE"
    # No defensible label exists (the subject-scale probe).
    UNLABELLED = "UNLABELLED"


@dataclass(frozen=True)
class Rung:
    kind: str
    level: float
    expected: ExpectedVerdict


# The ladder. Severities were chosen from the transform definitions, not from
# any measurement on this corpus.
DEGRADATION_LADDER: tuple[Rung, ...] = (
    # Defocus. Below sigma 1 a photograph is still usable for surface
    # inspection; at sigma 4 and beyond surface texture is gone.
    Rung("gaussian_blur", 1.0, ExpectedVerdict.BORDERLINE),
    Rung("gaussian_blur", 2.0, ExpectedVerdict.BORDERLINE),
    Rung("gaussian_blur", 4.0, ExpectedVerdict.UNUSABLE),
    Rung("gaussian_blur", 7.0, ExpectedVerdict.UNUSABLE),
    # Underexposure. Phase 1 measured that shadow clipping needs roughly gain
    # 0.05, so 0.15 is dark but recoverable and 0.06 is not.
    Rung("underexpose", 0.6, ExpectedVerdict.BORDERLINE),
    Rung("underexpose", 0.25, ExpectedVerdict.BORDERLINE),
    Rung("underexpose", 0.12, ExpectedVerdict.UNUSABLE),
    Rung("underexpose", 0.06, ExpectedVerdict.UNUSABLE),
    # Overexposure. convertScaleAbs saturates, so high gains genuinely destroy
    # highlight detail rather than merely brightening.
    Rung("overexpose", 1.6, ExpectedVerdict.BORDERLINE),
    Rung("overexpose", 2.6, ExpectedVerdict.UNUSABLE),
    Rung("overexpose", 3.5, ExpectedVerdict.UNUSABLE),
    # Contrast collapse toward flat grey.
    Rung("reduce_contrast", 0.5, ExpectedVerdict.BORDERLINE),
    Rung("reduce_contrast", 0.2, ExpectedVerdict.UNUSABLE),
    Rung("reduce_contrast", 0.08, ExpectedVerdict.UNUSABLE),
    # Specular glare over part of the frame.
    Rung("glare", 0.5, ExpectedVerdict.BORDERLINE),
    Rung("glare", 0.9, ExpectedVerdict.UNUSABLE),
    # Occlusion of the frame area.
    Rung("occlude", 0.2, ExpectedVerdict.BORDERLINE),
    Rung("occlude", 0.5, ExpectedVerdict.UNUSABLE),
    # Subject scale: the confound probe. Deliberately unlabelled - a small
    # subject is not a defective capture, it is a harder one, and pretending
    # otherwise would smuggle a judgement into the rubric.
    Rung("shrink_subject", 0.5, ExpectedVerdict.UNLABELLED),
    Rung("shrink_subject", 0.25, ExpectedVerdict.UNLABELLED),
)

# Undegraded real images. Track A survived a visual curation pass that removed
# obviously defective photographs, so presuming them acceptable is defensible
# and any block against them is a real false block. Track B was curated for
# subject identifiability only - those images are deliberately hard - so they
# carry no scored label.
BASE_EXPECTED: dict[str, ExpectedVerdict] = {
    "CLEAN_BASE": ExpectedVerdict.ACCEPTABLE,
    "NATURAL_SCENE": ExpectedVerdict.UNLABELLED,
}


def derived_path(derived_id: str, root: Path = DERIVED_DIR) -> Path:
    """Local location of a materialised derived sample, if one was written."""
    return root / f"{hashlib.sha256(derived_id.encode('utf-8')).hexdigest()[:20]}.png"


def regenerate(base_image: np.ndarray, transformation: str, level: float, seed: int) -> np.ndarray:
    """Recreate a derived sample from its base and its recorded parameters.

    Derived bytes are not kept: 1,300 losslessly encoded variants of 1920px
    photographs is several gigabytes of data that is fully determined by a base
    image, a transformation name, a level and a seed. Regenerating is cheaper
    than storing, and the manifest records each derived hash so a regenerated
    sample can be checked against the one that was measured rather than trusted.
    """
    degraded, _ = apply_degradation(
        base_image, DegradationSpec(kind=transformation, level=level, seed=seed)
    )
    return degraded


def derived_digest(image: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(image).tobytes() + str(image.shape).encode("ascii")
    ).hexdigest()


def build_derived_set(
    records: list[LicensedImageRecord],
    ladder: tuple[Rung, ...] = DEGRADATION_LADDER,
    seed: int = DEFAULT_SEED,
    root: Path = DERIVED_DIR,
    materialise: bool = False,
) -> dict:
    """Manifest every derived sample for the given base records.

    Bytes are written only when `materialise` is set, for inspecting a specific
    variant by hand. The measurement path regenerates instead - see `regenerate`.
    """
    if materialise:
        root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for base in records:
        image = load_image(base)
        for rung in ladder:
            spec = DegradationSpec(kind=rung.kind, level=rung.level, seed=seed)
            degraded, metadata = apply_degradation(image, spec)
            digest = derived_digest(degraded)
            record = derive_from(
                base,
                transformation=rung.kind,
                parameters={"level": rung.level},
                seed=seed,
                derived_sha256=digest,
            )
            if materialise:
                cv2.imwrite(str(derived_path(record.derived_id, root)), degraded)

            entries.append({
                **record.to_dict(),
                "expected_verdict": rung.expected.value,
                "degradation_metadata": metadata,
            })

    return {
        "derivation_version": DERIVATION_VERSION,
        "seed": seed,
        "base_image_count": len(records),
        "derived_count": len(entries),
        "bytes_materialised": materialise,
        "reproducibility": (
            "each derived sample is regenerated from its base image, "
            "transformation, level and seed; derived_sha256 pins the result"
        ),
        "ladder": [
            {"kind": rung.kind, "level": rung.level, "expected": rung.expected.value}
            for rung in ladder
        ],
        "rubric_discipline": (
            "expected verdicts were fixed from transformation severity before any "
            "metric was computed; BORDERLINE rungs are reported but never scored"
        ),
        "claim_boundary": (
            "controlled degradations of real photographs; not equivalent to real "
            "camera failure, which adds sensor noise, optical aberration and "
            "depth-dependent blur that these transforms do not reproduce"
        ),
        "derived": entries,
    }


def load_derived(path: Path = DERIVED_MANIFEST) -> list[dict]:
    if not path.is_file():
        raise CorpusError(
            f"{path.name} not found; run "
            "`python -m competition.evaluation.degrade_licensed_corpus` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))["derived"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.degrade_licensed_corpus",
        description="Controlled degradation of licensed real base images.",
    )
    parser.add_argument("--split", choices=["calibration", "held_out", "both"],
                        default="calibration")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--materialise", action="store_true",
                        help="also write the derived image bytes (git-ignored)")
    args = parser.parse_args(argv)

    stamped = split_records()
    wanted = (
        {Split.CALIBRATION.value, Split.HELD_OUT.value} if args.split == "both"
        else {args.split}
    )
    selected = [record for record in stamped if record.split in wanted]
    if not selected:
        raise SystemExit(f"no records in split {args.split!r}")

    document = build_derived_set(selected, seed=args.seed, materialise=args.materialise)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "bases": document["base_image_count"],
        "derived": document["derived_count"],
        "by_expected": dict(Counter(e["expected_verdict"] for e in document["derived"])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
