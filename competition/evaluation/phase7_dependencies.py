"""Phase 7: freeze what the deployed container actually contains.

Two artifacts, from one source of truth - the built image, not the requirements
file. A requirements file states an intention; the image states a fact, and a
submission that claims reproducibility should be checked against the fact.

The licence column is filled only where package metadata states a licence.
Where it does not, the field says so rather than guessing. An invented licence
in a bill of materials is worse than an absent one, because it looks like it was
checked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

DEPENDENCIES_VERSION = "phase7-dependencies-1.0.0"

RESULTS_DIR = Path("competition/evaluation/results/phase7")
FREEZE_PATH = RESULTS_DIR / "dependency_freeze.json"
SBOM_PATH = RESULTS_DIR / "sbom.json"

IMAGE = "agrivision:phase5"
DEPLOYED_DIGEST = (
    "sha256:b57eaf0506a243798ef21423048e1c571606d78243caa13bbf7d11b79d6ce7e7"
)

# What each direct dependency is actually for. Anything not listed here is
# present because something here pulled it in.
DIRECT_PURPOSE = {
    "opencv-python": "all perception: segmentation, quality metrics, cv2.dnn inference",
    "numpy": "array backing for every OpenCV operation",
    "pillow": "image decode fallback and format sniffing support",
    "fastapi": "HTTP API surface",
    "uvicorn": "ASGI server",
    "pydantic": "request and response schemas",
    "boto3": "S3 model artifact fetch, DynamoDB trace persistence",
    "python-multipart": "multipart upload parsing",
}

# Absent on purpose, each for a measured reason. Recorded so their absence reads
# as a decision rather than an oversight.
DELIBERATELY_ABSENT = {
    "torch": "ONNX export only; inference runs through cv2.dnn",
    "torchvision": "ONNX export only",
    "onnx": "export and parity checking only",
    "onnxruntime": "cv2.dnn reads the .onnx file directly",
    "matplotlib": "evaluation plots only, never served",
    "pytest": "test tooling",
    "httpx": "test tooling",
}


class DependencyError(RuntimeError):
    """Raised when the image cannot be inspected."""


def _docker(*args: str) -> str:
    result = subprocess.run(
        ["sudo", "-n", "docker", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise DependencyError(result.stderr.strip()[:300])
    return result.stdout


PROBE = r"""
import json, platform, sys
import importlib.metadata as md
packages = []
for dist in md.distributions():
    meta = dist.metadata
    classifiers = [c for c in (meta.get_all("Classifier") or [])
                   if c.startswith("License ::")]
    declared = (meta.get("License") or "").strip().replace("\n", " ")
    packages.append({
        "name": meta["Name"],
        "version": dist.version,
        "license_expression": (meta.get("License-Expression") or "").strip() or None,
        "license_declared": declared or None,
        "license_classifiers": classifiers or None,
    })
print(json.dumps({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": sorted(packages, key=lambda p: (p["name"] or "").lower()),
}))
"""


def inspect_image(image: str = IMAGE) -> dict:
    payload = _docker("run", "--rm", "--entrypoint", "python", image, "-c", PROBE)
    data = json.loads(payload)
    check = _docker("run", "--rm", "--entrypoint", "python", image, "-m", "pip", "check")
    data["pip_check"] = check.strip()
    data["pip_check_clean"] = "No broken requirements found" in check
    return data


def _licence_for(package: dict) -> dict:
    """Report a licence only where the package's own metadata states one."""
    # PEP 639 License-Expression first: it is the field packaging tools now
    # populate, and it carries an SPDX identifier rather than free text.
    expression = package.get("license_expression")
    if expression:
        return {"licence": expression,
                "source": "package metadata License-Expression (PEP 639, SPDX)"}
    declared = package.get("license_declared")
    classifiers = package.get("license_classifiers") or []
    if declared and len(declared) < 60:
        return {"licence": declared, "source": "package metadata License field"}
    if classifiers:
        return {"licence": classifiers[0].replace("License :: OSI Approved :: ", ""),
                "source": "package metadata Trove classifier"}
    if declared:
        return {"licence": None, "source": "License field present but not a short "
                                           "identifier; not interpreted"}
    return {"licence": None,
            "source": "package metadata declares no licence; not guessed"}


def build(image: str = IMAGE) -> tuple[dict, dict]:
    inspected = inspect_image(image)
    packages = inspected["packages"]

    freeze = {
        "dependencies_version": DEPENDENCIES_VERSION,
        "source_of_truth": (
            "The built container image, inspected from inside it. Not the "
            "requirements file, which states an intention rather than a fact."),
        "image": image,
        "deployed_image_digest": DEPLOYED_DIGEST,
        "python": inspected["python"],
        "platform": inspected["platform"],
        "package_count": len(packages),
        "pip_check": inspected["pip_check"],
        "pip_check_clean": inspected["pip_check_clean"],
        "pinned": {
            package["name"]: package["version"]
            for package in packages
            if package["name"].lower() in DIRECT_PURPOSE
        },
        "frozen": {package["name"]: package["version"] for package in packages},
        "deliberately_absent": DELIBERATELY_ABSENT,
        "research_contamination_check": {
            "clean": not ({p["name"].lower() for p in packages}
                          & set(DELIBERATELY_ABSENT)),
            "definition": (
                "No research or build-time dependency is present in the serving "
                "runtime. torch and torchvision together are the largest "
                "avoidable weight in an image of this kind."),
        },
    }

    sbom = {
        "dependencies_version": DEPENDENCIES_VERSION,
        "format": "lightweight dependency manifest, not CycloneDX or SPDX",
        "format_note": (
            "No SBOM generator is available in this environment, so this is "
            "produced from package metadata inside the image. It is honest about "
            "being lighter than a standard SBOM rather than claiming to be one."),
        "image": image,
        "deployed_image_digest": DEPLOYED_DIGEST,
        "python": inspected["python"],
        "licence_rule": (
            "Recorded only where the package's own metadata states it. Where "
            "metadata is unclear the licence is null and the reason is given; "
            "no value is inferred."),
        "components": [
            {
                "name": package["name"],
                "version": package["version"],
                **_licence_for(package),
                "purpose": DIRECT_PURPOSE.get(
                    package["name"].lower(), "transitive dependency"),
                "direct": package["name"].lower() in DIRECT_PURPOSE,
            }
            for package in packages
        ],
    }
    sbom["component_count"] = len(sbom["components"])
    sbom["direct_count"] = sum(1 for c in sbom["components"] if c["direct"])
    sbom["licence_unknown_count"] = sum(
        1 for c in sbom["components"] if c["licence"] is None)
    return freeze, sbom


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    freeze, sbom = build()
    FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SBOM_PATH.write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"python              : {freeze['python']}")
    print(f"packages in image   : {freeze['package_count']}")
    print(f"pip check           : {freeze['pip_check'].strip()}")
    print(f"research clean      : {freeze['research_contamination_check']['clean']}")
    print(f"SBOM components     : {sbom['component_count']} "
          f"({sbom['direct_count']} direct)")
    print(f"licence unknown     : {sbom['licence_unknown_count']} "
          f"(recorded as unknown, not guessed)")
    print(f"\nwritten: {FREEZE_PATH}\n         {SBOM_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
