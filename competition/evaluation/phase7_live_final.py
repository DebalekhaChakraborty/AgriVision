"""Phase 7: the last check on the endpoint judges will actually open.

Two jobs. The smoke run exercises the scenarios a judge is likely to try and
confirms each one still behaves as documented. The stability record captures
what is deployed right now - image digest, model checksum, backing services,
and whether any deployment is sitting in a failed state - so the submission
names a specific revision rather than "the live service".

Nothing here may change the system. If a scenario disagrees with the recorded
Phase 6 behaviour that is a finding to report, not a threshold to adjust.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LIVE_FINAL_VERSION = "phase7-live-final-1.0.0"

SERVICE_URL = "https://yp2ajauzkm.us-east-1.awsapprunner.com"
SERVICE_ARN = (
    "arn:aws:apprunner:us-east-1:341181499761:service/"
    "agrivision-inspection/9be40db3a6d147c79454d1e9c90fae08"
)
MODEL_BUCKET = "agrivision-model-artifacts-341181499761"
MODEL_KEY = "models/mobilenetv3_large_v2exp004/model.onnx"

RESULTS_DIR = Path("competition/evaluation/results/phase7")
SMOKE_PATH = RESULTS_DIR / "live_final_smoke.json"

# What each scenario must do. Taken from the recorded Phase 6 behaviour, not
# invented here, so a disagreement is visible rather than absorbed.
EXPECTED = {
    "NORMAL": {"model_invoked": True, "blocking": False},
    "UNDEREXPOSED": {"model_invoked": None, "blocking": None},
    "SEVERE_BLUR": {"model_invoked": False, "blocking": True},
    "LOW_CONTRAST": {"model_invoked": None, "blocking": None},
    "FOREGROUND_FAILURE": {"model_invoked": False, "blocking": True},
}

BLOCKING_TERMINALS = {
    "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "REQUEST_HUMAN_REVIEW",
    "FAILED_SAFE",
}


def _get(url: str, timeout: float = 90) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
            try:
                return response.status, json.loads(payload)
            except ValueError:
                return response.status, payload.decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def _post_png(image, name: str) -> tuple[int, dict, float]:
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"could not encode {name}")
    body = buffer.tobytes()
    boundary = "----agrivision-phase7-final"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{SERVICE_URL}/inspect", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return (response.status, json.loads(response.read()),
                    (time.perf_counter() - started) * 1000)
    except urllib.error.HTTPError as error:
        return (error.code, json.loads(error.read()),
                (time.perf_counter() - started) * 1000)


def build_scenarios() -> dict:
    from competition.evaluation.phase3_agent_scenarios import (
        darken, defocus, reference_capture, scene_without_a_subject,
    )
    from competition.vision.degradation import reduce_contrast

    base = reference_capture()
    return {
        "NORMAL": base,
        "UNDEREXPOSED": darken(base, 0.12),
        "SEVERE_BLUR": defocus(base, 21),
        "LOW_CONTRAST": reduce_contrast(base, 0.26),
        "FOREGROUND_FAILURE": scene_without_a_subject(),
    }


def smoke(frozen_fingerprints: dict) -> dict:
    health_status, health = _get(f"{SERVICE_URL}/health")
    ready_status, ready = _get(f"{SERVICE_URL}/ready")
    page_status, page = _get(f"{SERVICE_URL}/")
    css_status, _ = _get(f"{SERVICE_URL}/static/app.css")
    js_status, script = _get(f"{SERVICE_URL}/static/app.js")

    rows = []
    for name, image in build_scenarios().items():
        status, body, elapsed = _post_png(image, name)
        trace_ok = False
        steps = 0
        if status == 200 and body.get("run_id"):
            trace_status, trace = _get(
                f"{SERVICE_URL}/inspection/{body['run_id']}/trace")
            if trace_status == 200 and isinstance(trace, dict):
                steps = len(trace.get("trace", {}).get("steps", []))
                trace_ok = steps > 0
        expected = EXPECTED[name]
        terminal = body.get("final_status", "")
        blocking = terminal in BLOCKING_TERMINALS
        rows.append({
            "scenario": name,
            "http_status": status,
            "final_status": terminal,
            "condition_model_invoked": body.get("condition_model_invoked"),
            "remediation_action": body.get("remediation_action", ""),
            "requested_human_action": body.get("requested_human_action", ""),
            "lossless_transport": body.get("lossless_transport"),
            "trace_retrievable": trace_ok,
            "trace_steps": steps,
            "client_latency_ms": round(elapsed, 1),
            "policy_fingerprints_match_frozen": (
                body.get("policy_fingerprints") == frozen_fingerprints),
            "matches_expected_model_invocation": (
                expected["model_invoked"] is None
                or body.get("condition_model_invoked") == expected["model_invoked"]),
            "matches_expected_blocking": (
                expected["blocking"] is None or blocking == expected["blocking"]),
            "run_id": body.get("run_id", ""),
        })

    return {
        "endpoints": {
            "health": {"status": health_status, "ok": health_status == 200,
                       "body": health},
            "ready": {"status": ready_status, "ok": ready_status == 200
                      and isinstance(ready, dict) and ready.get("ready") is True,
                      "body": ready},
            "page": {"status": page_status, "ok": page_status == 200,
                     "bytes": len(page) if isinstance(page, str) else 0},
            "app_css": {"status": css_status, "ok": css_status == 200},
            "app_js": {"status": js_status, "ok": js_status == 200},
        },
        "ui_surface": {
            "responsible_use_present": isinstance(page, str)
            and "Responsible use" in page,
            "one_primary_fruit_contract_present": isinstance(page, str)
            and "one primary fruit" in page.lower(),
            "no_food_safety_claim": isinstance(page, str)
            and "does not determine whether food" in page,
            "counterfactual_control_present": isinstance(script, str)
            and "counterfactual" in script.lower(),
            "raw_trace_disclosure_present": isinstance(page, str)
            and "rawTrace" in page,
        },
        "scenarios": rows,
        "all_http_200": all(row["http_status"] == 200 for row in rows),
        "all_traces_retrievable": all(row["trace_retrievable"] for row in rows),
        "all_fingerprints_match": all(
            row["policy_fingerprints_match_frozen"] for row in rows),
        "all_expectations_met": all(
            row["matches_expected_model_invocation"]
            and row["matches_expected_blocking"] for row in rows),
    }


def stability(runner: str) -> dict:
    def aws(*args: str) -> dict | None:
        result = subprocess.run([runner, "aws", *args],
                                capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout or "{}")
        except ValueError:
            return None

    service = (aws("apprunner", "describe-service", "--service-arn", SERVICE_ARN,
                   "--output", "json") or {}).get("Service", {})
    operations = (aws("apprunner", "list-operations", "--service-arn", SERVICE_ARN,
                      "--output", "json") or {}).get("OperationSummaryList", [])
    image = aws("ecr", "describe-images", "--repository-name", "agrivision",
                "--image-ids", "imageTag=phase4", "--output", "json") or {}
    details = (image.get("imageDetails") or [{}])[0]
    model_object = aws("s3api", "head-object", "--bucket", MODEL_BUCKET,
                       "--key", MODEL_KEY, "--output", "json")

    failed = [
        {"type": op.get("Type"), "status": op.get("Status"),
         "started": str(op.get("StartedAt"))}
        for op in operations
        if op.get("Status") in {"FAILED", "ROLLBACK_FAILED"}
    ]
    latest = operations[0] if operations else {}

    return {
        "FINAL_DEMO_REVISION": {
            "service_url": SERVICE_URL,
            "service_id": service.get("ServiceId"),
            "service_status": service.get("Status"),
            "image_digest": details.get("imageDigest"),
            "image_tag_resolved": "phase4",
            "image_bytes": details.get("imageSizeInBytes"),
            "image_pushed_at": str(details.get("imagePushedAt")),
            "last_operation": {
                "id": latest.get("Id"), "type": latest.get("Type"),
                "status": latest.get("Status"),
                "ended": str(latest.get("EndedAt"))},
            "instance_role": service.get("InstanceConfiguration", {}).get(
                "InstanceRoleArn", "").rsplit("/", 1)[-1],
            "deployment_principal": "agrivision-deployer (scoped IAM user)",
            "audit_principal": "agrivision-auditor (read-only, logs only)",
            "root_required": False,
        },
        "service_running": service.get("Status") == "RUNNING",
        "no_pending_failed_deployment": not failed,
        "failed_operations": failed,
        "model_object_available": model_object is not None,
        "model_object_bytes": (model_object or {}).get("ContentLength"),
        "tag_mutability_note": (
            "The tag is a moving pointer; the digest is the identity. Both are "
            "recorded so a reader can tell which was resolved."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner")
    args = parser.parse_args(argv)

    frozen = json.loads(
        Path("competition/evaluation/results/phase6/frozen_system.json")
        .read_text(encoding="utf-8"))
    fingerprints = frozen["policy"]["fingerprints"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "live_final_version": LIVE_FINAL_VERSION,
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_fingerprint": frozen["frozen_fingerprint"],
        "smoke": smoke(fingerprints),
        "stability": stability(args.runner),
        "policy_unchanged_note": (
            "This run verifies behaviour. Nothing in the policy, the thresholds "
            "or the model was modified in response to it."),
    }
    SMOKE_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    smoke_block = document["smoke"]
    for name, endpoint in smoke_block["endpoints"].items():
        print(f"  {'ok ' if endpoint['ok'] else 'FAIL'} {name} ({endpoint['status']})")
    print()
    for row in smoke_block["scenarios"]:
        mark = "ok " if (row["matches_expected_model_invocation"]
                         and row["matches_expected_blocking"]
                         and row["trace_retrievable"]) else "CHECK"
        print(f"  {mark} {row['scenario']:20} {row['final_status']:18} "
              f"model={str(row['condition_model_invoked']):5} "
              f"trace={row['trace_steps']:2} {row['client_latency_ms']:7.0f} ms")
    stab = document["stability"]
    print(f"\nservice        : {stab['FINAL_DEMO_REVISION']['service_status']}")
    print(f"image digest   : {stab['FINAL_DEMO_REVISION']['image_digest']}")
    print(f"failed deploys : {len(stab['failed_operations'])}")
    print(f"model object   : {stab['model_object_available']} "
          f"({stab['model_object_bytes']} bytes)")
    print(f"fingerprints   : all match = {smoke_block['all_fingerprints_match']}")
    print(f"\nwritten: {SMOKE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
