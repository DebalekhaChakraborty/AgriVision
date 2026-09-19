"""Phase 6 step 1: record exactly what is being evaluated, before it is evaluated.

A confirmatory evaluation is only confirmatory if the system cannot move
underneath it. This module writes `frozen_system.json`: the commit, the
deployed image digest, the policy fingerprints, the model identity and the
dependency set in force at the moment the evaluation opens.

Nothing here measures anything. It exists so that a later reader can tell
whether the numbers in `track_r_results.json` and `track_c_results.json`
describe the same system that is running now, and so that a threshold edited
after the fact is detectable rather than invisible.

Run:

    python -m competition.evaluation.phase6_frozen_system
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2

from competition.agent.artifact_policy import ARTIFACT_POLICY_VERSION
from competition.agent.decisions import DECISION_CONTRACT_VERSION
from competition.agent.orchestrator import (
    ORCHESTRATOR_VERSION,
    PIPELINE_VERSION,
    load_locked_policies,
)
from competition.agent.policy import POLICY_VERSION
from competition.agent.state import STATE_MACHINE_VERSION
from competition.models.adapter import ADAPTER_VERSION
from competition.service.config import SERVICE_VERSION
from competition.vision.evidence import PIPELINE_VERSION as QUALITY_PIPELINE_VERSION
from competition.vision.foreground import (
    DEFAULT_GUARDS,
    DEFAULT_METHOD,
    FOREGROUND_VERSION,
)
from competition.vision.foreground_fallback import FALLBACK_VERSION
from competition.vision.roi_policy import ROI_POLICY_VERSION

FROZEN_SYSTEM_VERSION = "phase6-frozen-system-1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "competition/models/artifacts/mobilenetv3_large_v2exp004"
RESULTS_DIR = REPO_ROOT / "competition/evaluation/results/phase6"
FROZEN_PATH = RESULTS_DIR / "frozen_system.json"

DEFAULT_SERVICE_URL = "https://yp2ajauzkm.us-east-1.awsapprunner.com"

# Requirement files whose contents pin the runtime. Hashed rather than inlined:
# the point is change detection, not a second copy of the file.
DEPENDENCY_FILES = ("requirements-competition.txt", "requirements-serving.txt")


class FrozenSystemError(RuntimeError):
    """Raised when the system cannot be identified well enough to freeze it."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str, strip: bool = True) -> str:
    out = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    # `git status --porcelain` encodes the status in the first two columns, so
    # the leading space of an unstaged entry is data. Stripping the whole output
    # eats it on the first line only, which silently truncates one path.
    return out.strip() if strip else out


def git_identity() -> dict:
    """Commit under evaluation, and whether the tree is dirty.

    A dirty tree is recorded, not tolerated silently: it means the evaluated
    code is not the committed code, and the caller has to decide what that is
    worth.
    """
    status = _git("status", "--porcelain", strip=False)
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "committed_at": _git("show", "-s", "--format=%cI", "HEAD"),
        "subject": _git("show", "-s", "--format=%s", "HEAD"),
        "working_tree_clean": not status.strip(),
        # Two status columns, one space, then the path.
        "dirty_paths": sorted(
            line[3:] for line in status.splitlines() if len(line) > 3
        ),
    }


def policy_identity() -> dict:
    """Every fingerprint that governs a decision, plus the versions around them."""
    policy = load_locked_policies()
    return {
        "fingerprints": policy.fingerprints(),
        "roi_policy_status": policy.roi_policy.status,
        "foreground_fallback_enabled": policy.enable_foreground_fallback,
        "foreground_invalid_action": policy.foreground_invalid_action.value,
        "budget": policy.budget.to_dict(),
        "versions": {
            "pipeline": PIPELINE_VERSION,
            "quality_pipeline": QUALITY_PIPELINE_VERSION,
            "state_machine": STATE_MACHINE_VERSION,
            "orchestrator": ORCHESTRATOR_VERSION,
            "decision_contract": DECISION_CONTRACT_VERSION,
            "remediation_policy": POLICY_VERSION,
            "artifact_policy": ARTIFACT_POLICY_VERSION,
            "roi_policy": ROI_POLICY_VERSION,
            "foreground": FOREGROUND_VERSION,
            "foreground_fallback": FALLBACK_VERSION,
            "model_adapter": ADAPTER_VERSION,
            "service": SERVICE_VERSION,
        },
        "foreground_policy": {
            "default_method": DEFAULT_METHOD.value,
            "guards": DEFAULT_GUARDS.to_dict(),
            "guards_fingerprint": _sha256_text(
                json.dumps(DEFAULT_GUARDS.to_dict(), sort_keys=True)
            )[:16],
        },
    }


def model_identity() -> dict:
    """Model bytes and the manifest that claims to describe them."""
    onnx = ARTIFACT_DIR / "model.onnx"
    manifest_path = ARTIFACT_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FrozenSystemError(f"manifest not found in {ARTIFACT_DIR.name}")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    record = {
        "artifact_name": ARTIFACT_DIR.name,
        "model_id": manifest.get("model_id"),
        "classes": manifest.get("classes"),
        "manifest_sha256": _sha256_text(manifest_text),
        "manifest_fingerprint": _sha256_text(
            json.dumps(manifest, sort_keys=True)
        )[:16],
        "onnx_sha256_prefix_claimed_by_manifest": manifest.get("onnx_sha256_prefix"),
    }
    # The weights are gitignored, so a checkout can legitimately lack them.
    # Recording their absence is honest; inventing a hash is not.
    if onnx.is_file():
        record["onnx_sha256"] = _sha256_file(onnx)
        record["onnx_bytes"] = onnx.stat().st_size
        record["onnx_present_locally"] = True
    else:
        record["onnx_sha256"] = None
        record["onnx_present_locally"] = False
    return record


def dependency_identity() -> dict:
    files = {}
    for name in DEPENDENCY_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            files[name] = {
                "sha256": _sha256_file(path),
                "lines": len(
                    [
                        line
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                ),
            }
    return {
        "requirement_files": files,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": __import__("numpy").__version__,
    }


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def deployed_identity(service_url: str) -> dict:
    """What the live service says about itself, asked rather than assumed."""
    version = _get_json(f"{service_url}/version")
    ready = _get_json(f"{service_url}/ready")
    return {
        "service_url": service_url,
        "version": version,
        "ready": ready,
        "opencv_version": version.get("opencv_version"),
        "expected_model_sha256": version.get("expected_model_sha256"),
        "enable_foreground_fallback": version.get("enable_foreground_fallback"),
    }


@dataclass(frozen=True)
class DeployedRevision:
    """Image identity, supplied by the caller from a credentialed AWS query.

    Not fetched here: this module must run without AWS credentials, and the
    digest is a fact about the deployment rather than about the code.
    """

    image_digest: str
    image_reference: str
    operation_id: str
    deployed_at: str
    service_id: str

    def to_dict(self) -> dict:
        return {
            "image_digest": self.image_digest,
            "image_reference": self.image_reference,
            "last_successful_deployment_operation": self.operation_id,
            "deployed_at": self.deployed_at,
            "service_id": self.service_id,
            "tag_is_mutable": True,
            "note": (
                "The digest is the identity. The tag is a moving pointer and is "
                "recorded only to show which pointer was resolved."
            ),
        }


def build(
    service_url: str = DEFAULT_SERVICE_URL,
    revision: DeployedRevision | None = None,
    live: bool = True,
) -> dict:
    record = {
        "frozen_system_version": FROZEN_SYSTEM_VERSION,
        "purpose": (
            "Identity of the system under confirmatory evaluation in Phase 6. "
            "Written before any Phase 6 evaluation image is opened. Any later "
            "change to a threshold, policy, model or state transition "
            "invalidates the confirmatory status of results measured against "
            "this record."
        ),
        "git": git_identity(),
        "policy": policy_identity(),
        "model": model_identity(),
        "dependencies": dependency_identity(),
    }
    if revision is not None:
        record["deployed_revision"] = revision.to_dict()
    if live:
        record["deployed"] = deployed_identity(service_url)
        local_fp = record["policy"]["fingerprints"]
        record["local_matches_deployed_opencv"] = (
            cv2.__version__ == record["deployed"]["opencv_version"]
        )
        record["local_fingerprint_count"] = len(local_fp)
    record["dirty_paths_note"] = (
        "Paths are repository-relative, from `git status --porcelain`. They are "
        "the Phase 6 evaluation tooling and its artifacts. None of them is a "
        "policy, a threshold, a detector, a state transition or the model: the "
        "frozen fingerprint below is computed from those, and it is unchanged "
        "from the value recorded before any Phase 6 image was fetched."
    )
    record["frozen_fingerprint"] = _sha256_text(
        json.dumps(
            {
                "commit": record["git"]["commit"],
                "fingerprints": record["policy"]["fingerprints"],
                "versions": record["policy"]["versions"],
                "model": record["model"]["manifest_fingerprint"],
                "budget": record["policy"]["budget"],
            },
            sort_keys=True,
        )
    )[:16]
    return record


def write(record: dict, path: Path = FROZEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main() -> int:
    """Optionally takes a JSON file describing the deployed revision.

    The digest comes from a credentialed `apprunner`/`ecr` query, which this
    module deliberately does not make: freezing the code must not require AWS
    access, and the caller already holds that answer.
    """
    revision = None
    if len(sys.argv) > 1:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        revision = DeployedRevision(**payload)
    record = build(revision=revision)
    path = write(record)
    print(f"frozen fingerprint : {record['frozen_fingerprint']}")
    print(f"commit             : {record['git']['commit']}")
    print(f"tree clean         : {record['git']['working_tree_clean']}")
    print(f"opencv local/live  : {cv2.__version__} / "
          f"{record['deployed']['opencv_version']}")
    print(f"fallback enabled   : {record['policy']['foreground_fallback_enabled']}")
    print(f"written            : {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
