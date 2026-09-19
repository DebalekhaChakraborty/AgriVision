"""Phase 6 execution: run the frozen deployed service over both tracks.

The headline Phase 6 numbers come from the deployed App Runner service, not from
a local pipeline. A local run would measure the code; this measures the thing a
judge can actually open, including its transport, its model artifact and its
persistence.

Nothing here computes a metric or makes a judgement. It collects raw responses
and traces, records the policy fingerprints the service reports, and stops. The
scoring lives in `phase6_metrics`, so that collection and interpretation cannot
be quietly entangled.

Run:

    python -m competition.evaluation.phase6_run r      # Track R, natural images
    python -m competition.evaluation.phase6_run c      # Track C, controlled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RUN_VERSION = "phase6-run-1.0.0"

SERVICE_URL = "https://yp2ajauzkm.us-east-1.awsapprunner.com"
RESULTS_DIR = Path("competition/evaluation/results/phase6")
TRACK_R_RAW = RESULTS_DIR / "track_r_raw.json"
TRACK_C_RAW = RESULTS_DIR / "track_c_raw.json"

REQUEST_TIMEOUT_S = 240.0
ATTEMPTS = 3
BACKOFF_S = 5.0

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class RunError(RuntimeError):
    """Raised when the deployed service cannot be exercised as required."""


def _multipart(payload: bytes, filename: str, media_type: str) -> tuple[bytes, str]:
    boundary = "----agrivision-phase6"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def post_inspection(path: Path, base_url: str = SERVICE_URL) -> tuple[int, dict, float]:
    """One /inspect call, timed from the client. Retries transport failures only.

    An HTTP error carrying a JSON body is a decision by the service and is
    returned as-is; retrying it would just ask the same question again.
    """
    payload = path.read_bytes()
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise RunError(f"unsupported extension {path.suffix!r}")
    body, content_type = _multipart(payload, path.name, media_type)

    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(
            f"{base_url}/inspect", data=body, headers={"Content-Type": content_type}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                elapsed = (time.perf_counter() - started) * 1000.0
                return response.status, json.loads(response.read()), elapsed
        except urllib.error.HTTPError as error:
            elapsed = (time.perf_counter() - started) * 1000.0
            try:
                return error.code, json.loads(error.read()), elapsed
            except Exception:  # noqa: BLE001
                return error.code, {"error": "unparseable"}, elapsed
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt < ATTEMPTS - 1:
                time.sleep(BACKOFF_S * (attempt + 1))
    raise RunError(f"{path.name}: service unreachable after {ATTEMPTS} attempts: {last}")


def fetch_trace(run_id: str, base_url: str = SERVICE_URL) -> dict | None:
    """Retrieve the persisted trace, proving the write actually round-tripped."""
    try:
        with urllib.request.urlopen(
            f"{base_url}/inspection/{run_id}/trace", timeout=90
        ) as response:
            return json.loads(response.read())
    except Exception:  # noqa: BLE001
        return None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(label: dict, path: Path, status: int, body: dict, ms: float) -> dict:
    """Everything worth keeping about one inspection, and nothing else.

    No filesystem path is recorded: the id plus the variant reconstructs it, and
    a committed artifact must not carry local paths.
    """
    trace = fetch_trace(body["run_id"]) if status == 200 and body.get("run_id") else None
    steps = (trace or {}).get("trace", {}).get("steps", [])
    quality = body.get("quality_summary") or {}
    artifact = body.get("artifact_summary") or {}
    return {
        **label,
        "http_status": status,
        "transport_sha256": _digest(path),
        "transport_media_type": MEDIA_TYPES[path.suffix.lower()],
        "client_latency_ms": round(ms, 1),
        "run_id": body.get("run_id", ""),
        "final_status": body.get("final_status", ""),
        "requested_human_action": body.get("requested_human_action", ""),
        "advisory_human_action": artifact.get("advisory_human_action", ""),
        "terminal_reason_code": body.get("terminal_reason_code", ""),
        "foreground_valid": quality.get("foreground_valid"),
        "foreground_invalid_reasons": quality.get("foreground_invalid_reasons", []),
        "quality_flags": quality.get("quality_flags", []),
        "artifact_flags": artifact.get("artifact_flags", []),
        "advisory_flags": artifact.get("advisory_flags", []),
        "remediation_applied": body.get("remediation_applied"),
        "remediation_accepted": body.get("remediation_accepted"),
        "remediation_action": body.get("remediation_action", ""),
        "condition_model_invoked": body.get("condition_model_invoked"),
        "condition_evidence": body.get("condition_evidence"),
        "model_confidence": body.get("model_confidence"),
        "lossless_transport": body.get("lossless_transport"),
        "original_image_sha256": body.get("original_image_sha256", ""),
        "canonical_image_sha256": body.get("canonical_image_sha256", ""),
        "policy_fingerprints": body.get("policy_fingerprints", {}),
        "pipeline_version": body.get("pipeline_version", ""),
        "opencv_version": body.get("opencv_version", ""),
        "state_path": body.get("state_path", []),
        "trace_retrievable": bool(steps),
        "trace_step_count": len(steps),
        "trace_id": body.get("trace_id", ""),
        "error": body.get("error", "") if status != 200 else "",
        "reason_code": body.get("reason_code", "") if status != 200 else "",
    }


def _envelope(track: str, observations: list[dict], extra: dict) -> dict:
    return {
        "run_version": RUN_VERSION,
        "track": track,
        "service_url": SERVICE_URL,
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_target": "DEPLOYED AWS APP RUNNER SERVICE",
        "count": len(observations),
        **extra,
        "observations": observations,
    }


def run_track_r() -> dict:
    from competition.evaluation.phase6_source_pool import PHASE6_RAW, load_manifest

    manifest = load_manifest()
    observations = []
    for index, record in enumerate(manifest["records"], start=1):
        path = PHASE6_RAW / f"{record['image_id']}{record['file_extension']}"
        status, body, ms = post_inspection(path)
        observations.append(_observation(
            {"image_id": record["image_id"], "fruit_type": record["fruit_type"],
             "source_group_id": record["source_group_id"],
             "track_label": record["track"],
             "manifest_sha256": record["content_sha256"]},
            path, status, body, ms))
        print(f"  [{index:2}/{len(manifest['records'])}] {record['image_id']} "
              f"{record['fruit_type']:6} -> {body.get('final_status', status)}",
              file=sys.stderr)
    return _envelope("R", observations, {
        "purpose": "Natural-domain behaviour of the frozen system on fresh real images.",
        "transport_note": (
            "Images are sent in the format they were harvested in, which is what "
            "a natural client would send. `lossless_transport` records which."),
        "source_manifest_fingerprint": manifest["manifest_fingerprint"],
    })


def run_track_c() -> dict:
    from competition.evaluation.phase6_track_c import (
        EXPECTED_ACTIONS,
        VARIANT_DIR,
        VARIANT_KEYS,
    )

    prereg = json.loads(EXPECTED_ACTIONS.read_text(encoding="utf-8"))
    scenarios = prereg["scenarios"]
    observations = []
    for index, scenario in enumerate(scenarios, start=1):
        path = VARIANT_DIR / f"{scenario['base_id']}__{scenario['variant']}.png"
        if not path.is_file():
            raise RunError(f"variant not materialised: {path.name}")
        status, body, ms = post_inspection(path)
        observations.append(_observation(
            {"base_id": scenario["base_id"], "variant": scenario["variant"],
             "fruit_type": scenario["fruit_type"],
             "expected_first_action": scenario["expected_first_action"],
             "condition_model_must_run": scenario["condition_model_must_run"],
             "expected_terminal": scenario["expected_terminal"]},
            path, status, body, ms))
        print(f"  [{index:2}/{len(scenarios)}] {scenario['base_id']} "
              f"{scenario['variant']:26} -> {body.get('final_status', status)}",
              file=sys.stderr)
    return _envelope("C", observations, {
        "purpose": "Agent routing under controlled conditions on real bases.",
        "claim_boundary": prereg["claim_boundary"],
        "expected_actions_fingerprint": prereg["expected_actions_fingerprint"],
        "transport_note": "PNG throughout, so the controlled pixels survive transport.",
        "variant_keys": list(VARIANT_KEYS),
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("r", "c"))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.track == "r":
        document, path = run_track_r(), TRACK_R_RAW
    else:
        document, path = run_track_c(), TRACK_C_RAW
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{document['count']} observations -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
