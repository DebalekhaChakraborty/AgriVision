"""Phase 4: the HTTP surface, its limits, and what it refuses to leak."""

from __future__ import annotations

import dataclasses
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from competition.evaluation.phase3_agent_scenarios import (
    darken,
    defocus,
    reference_capture,
    scene_without_a_subject,
)
from competition.service.api import CLAIM_BOUNDARY, create_app
from competition.service.artifacts import ensure_model_artifact
from competition.service.config import EXPECTED_MODEL_SHA256, ServiceConfig
from competition.service.observability import JsonFormatter, Metrics, _scrub
from competition.service.persistence import InMemoryStore, WriteOutcome
from competition.service.uploads import UploadRejected, validate_upload

ARTIFACT_DIR = "competition/models/artifacts/mobilenetv3_large_v2exp004"


def encode(image: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


@pytest.fixture(scope="module")
def config() -> ServiceConfig:
    return ServiceConfig(model_dir=ARTIFACT_DIR)


@pytest.fixture(scope="module")
def model_available(config) -> bool:
    return ensure_model_artifact(config).usable


@pytest.fixture(scope="module")
def client(config, model_available):
    if not model_available:
        pytest.skip("model artifact unavailable in this checkout")
    app = create_app(config)
    with TestClient(app) as test_client:
        # Initialisation is asynchronous so the port opens immediately; a test
        # that asserts on readiness has to wait for it deliberately.
        assert app.state.service.wait_until_ready(60), "service never became ready"
        yield test_client


# --- 1/2/3: health and readiness ----------------------------------------------


def test_health_is_alive_immediately_before_initialisation_finishes():
    """The port must open before the model is fetched.

    Regression guard for a real App Runner deployment failure: doing the S3
    download inside the lifespan blocked uvicorn from accepting connections,
    the platform health check failed, and the deployment was reported as a
    broken image with no application logs at all.
    """
    slow = ServiceConfig(model_dir="/nonexistent/path")
    app = create_app(slow)
    with TestClient(app) as probe:
        assert probe.get("/health").status_code == 200


def test_health_is_alive_without_a_model():
    """Liveness must not depend on the model: a crash-looping container is worse."""
    broken = ServiceConfig(model_dir="/nonexistent/path")
    with TestClient(create_app(broken)) as probe:
        response = probe.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_is_false_without_a_model():
    broken = ServiceConfig(model_dir="/nonexistent/path")
    app = create_app(broken)
    with TestClient(app) as probe:
        app.state.service.wait_until_ready(30)
        response = probe.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["model_artifact_present"] is False


def test_ready_is_true_with_a_verified_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model_artifact_verified"] is True
    assert body["opencv_version"].startswith("5.")


# --- 11: checksum mismatch fails readiness ------------------------------------


def test_a_checksum_mismatch_prevents_readiness(model_available):
    """The property that matters most in the artifact path.

    Needs the artifact present: the claim is that a *mismatched* checksum
    blocks readiness, which is a different statement from a missing file
    blocking it. Without the bytes there is nothing to mismatch, so this skips
    rather than failing in a checkout that has not fetched the model.
    """
    if not model_available:
        pytest.skip("model artifact unavailable in this checkout")
    wrong = ServiceConfig(model_dir=ARTIFACT_DIR, expected_model_sha256="0" * 64)
    status = ensure_model_artifact(wrong)
    assert status.present is True
    assert status.verified is False
    assert status.usable is False
    app = create_app(wrong)
    with TestClient(app) as probe:
        app.state.service.wait_until_ready(30)
        response = probe.get("/ready")
    assert response.status_code == 503
    assert response.json()["model_artifact_verified"] is False


def test_the_expected_hash_is_the_real_artifact_hash(config, model_available):
    if not model_available:
        pytest.skip("model artifact unavailable")
    status = ensure_model_artifact(config)
    assert status.observed_sha256 == EXPECTED_MODEL_SHA256


def test_inspect_refuses_when_not_ready():
    broken = ServiceConfig(model_dir="/nonexistent/path")
    app = create_app(broken)
    with TestClient(app) as probe:
        app.state.service.wait_until_ready(30)
        response = probe.post(
            "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
        )
    assert response.status_code == 503
    assert response.json()["detail"]["reason_code"] == "MODEL_UNAVAILABLE"


# --- 4/5/6: input validation ---------------------------------------------------


def test_a_non_image_is_rejected(client):
    response = client.post(
        "/inspect", files={"image": ("notes.jpg", b"this is plain text", "image/jpeg")}
    )
    assert response.status_code == 415
    assert response.json()["detail"]["reason_code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_a_declared_mime_type_does_not_override_content(client):
    """A filename and a Content-Type are claims, not facts."""
    response = client.post(
        "/inspect", files={"image": ("real.png", b"%PDF-1.7 not an image", "image/png")}
    )
    assert response.status_code == 415


def test_a_corrupt_image_is_rejected(client):
    """Begins with the JPEG magic bytes and is not decodable."""
    response = client.post(
        "/inspect", files={"image": ("x.jpg", b"\xff\xd8\xff" + b"\x00" * 400, "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "UNDECODABLE_IMAGE"


def test_an_oversized_upload_is_rejected(client):
    payload = b"\xff\xd8\xff" + b"\x00" * (13 * 1024 * 1024)
    response = client.post(
        "/inspect", files={"image": ("big.jpg", payload, "image/jpeg")}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["reason_code"] == "UPLOAD_TOO_LARGE"


def test_an_empty_upload_is_rejected(client):
    response = client.post("/inspect", files={"image": ("e.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_a_tiny_image_is_rejected(config):
    with pytest.raises(UploadRejected) as error:
        validate_upload(encode(np.full((8, 8, 3), 120, np.uint8), ".png"), config)
    assert error.value.reason_code == "IMAGE_TOO_SMALL"


# --- 7/8: inspection outcomes --------------------------------------------------


def test_inspect_returns_a_structured_agent_result(client):
    response = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["final_status"] == "COMPLETE"
    assert body["condition_model_invoked"] is True
    assert len(body["original_image_sha256"]) == 64
    assert body["policy_fingerprints"]
    assert body["claim_boundary"] == CLAIM_BOUNDARY


def test_a_recapture_is_an_application_outcome_not_a_server_error(client):
    """The distinction the status-code semantics exist for."""
    response = client.post(
        "/inspect",
        files={"image": ("b.jpg", encode(defocus(reference_capture(), 21)), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["final_status"] == "REQUEST_RECAPTURE"
    assert body["requested_human_action"] == "REQUEST_RECAPTURE"
    assert body["condition_model_invoked"] is False
    assert body["condition_evidence"] is None


def test_segmentation_failure_is_also_a_200_with_an_action(client):
    response = client.post(
        "/inspect",
        files={"image": ("c.jpg", encode(scene_without_a_subject()), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["requested_human_action"]
    assert body["quality_summary"]["foreground_valid"] is False
    assert body["condition_model_invoked"] is False


def test_remediation_is_reported(client):
    response = client.post(
        "/inspect",
        files={"image": ("d.jpg", encode(darken(reference_capture(), 0.12)), "image/jpeg")},
    )
    body = response.json()
    assert body["remediation_applied"] is True
    assert body["canonical_image_sha256"] != body["original_image_sha256"]


def test_the_input_contract_is_stated_in_every_response(client):
    response = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    )
    contract = response.json()["deployment_input_contract"]
    assert "one primary fruit" in contract.lower()
    assert "not a food-safety statement" in contract.lower()


# --- 9/10: trace retrieval and privacy ----------------------------------------


def test_a_trace_is_retrievable_by_run_id(client):
    created = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    ).json()
    response = client.get(f"/inspection/{created['run_id']}/trace")
    assert response.status_code == 200
    trace = response.json()["trace"]
    assert trace["steps"]
    assert [s["step_id"] for s in trace["steps"]] == list(
        range(1, len(trace["steps"]) + 1))


def test_the_retrieved_trace_preserves_the_causal_chain(client):
    """A judge must see evidence -> decision -> next action, not a summary."""
    created = client.post(
        "/inspect",
        files={"image": ("b.jpg", encode(defocus(reference_capture(), 21)), "image/jpeg")},
    ).json()
    trace = client.get(f"/inspection/{created['run_id']}/trace").json()["trace"]
    quality = next(s for s in trace["steps"] if s["tool_name"] == "assess_capture_quality")
    assert "high_frequency_ratio" in quality["evidence_summary"]
    assert "focus_floor" in quality["evidence_summary"]
    decision = next(s for s in trace["steps"] if s["selected_action"])
    assert decision["decision_reason"]
    assert decision["evidence_maturity"]


def test_a_trace_contains_no_image_bytes_or_local_paths(client):
    created = client.post(
        "/inspect", files={"image": ("secret_photo.jpg", encode(reference_capture()),
                                     "image/jpeg")}
    ).json()
    payload = json.dumps(client.get(f"/inspection/{created['run_id']}/trace").json())
    for marker in ("/home/", "/Users/", "/tmp/", "C:\\", ".jpg", ".png",
                   "secret_photo"):
        assert marker not in payload, f"{marker!r} leaked into the trace"


def test_the_run_id_is_not_derived_from_the_filename(client):
    body = client.post(
        "/inspect", files={"image": ("identifiable-name.jpg", encode(reference_capture()),
                                     "image/jpeg")}
    ).json()
    assert "identifiable" not in body["run_id"]
    assert len(body["run_id"]) == 32


def test_an_unknown_run_id_is_404(client):
    assert client.get("/inspection/" + "0" * 32).status_code == 404


def test_a_malformed_run_id_is_400(client):
    assert client.get("/inspection/..%2Fetc%2Fpasswd").status_code in (400, 404)


# --- 12: persistence failures fail safely --------------------------------------


class BrokenStore(InMemoryStore):
    backend = "broken"

    def put(self, run_id, record):
        return WriteOutcome(stored=False, backend="broken", detail="SimulatedFailure")

    def get(self, run_id):
        raise RuntimeError("simulated store outage")


def test_a_persistence_write_failure_does_not_change_the_result(config, model_available):
    """The inspection already happened correctly; losing the record must not alter it."""
    if not model_available:
        pytest.skip("model artifact unavailable")
    app = create_app(config)
    with TestClient(app) as probe:
        assert app.state.service.wait_until_ready(60)
        app.state.service.store = BrokenStore()
        response = probe.post(
            "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
        )
    assert response.status_code == 200
    assert response.json()["final_status"] == "COMPLETE"
    assert response.headers["X-Trace-Stored"] == "false"


def test_a_persistence_read_failure_is_503_not_a_fabricated_trace(config, model_available):
    if not model_available:
        pytest.skip("model artifact unavailable")
    app = create_app(config)
    with TestClient(app) as probe:
        assert app.state.service.wait_until_ready(60)
        app.state.service.store = BrokenStore()
        response = probe.get("/inspection/" + "a" * 32)
    assert response.status_code == 503
    assert response.json()["detail"]["reason_code"] == "PERSISTENCE_ERROR"


# --- 13/14: structured logging -------------------------------------------------


def test_logs_are_json_with_the_competition_fields():
    import logging

    record = logging.LogRecord("agrivision", logging.INFO, "", 0, "agent_step", (), None)
    record.fields = {"run_id": "r1", "action": "REQUEST_RECAPTURE",
                     "tool": "request_recapture", "state": "REQUEST_RECAPTURE",
                     "evidence_maturity": "CALIBRATED", "duration_ms": 5.0}
    payload = json.loads(JsonFormatter().format(record))
    for key in ("run_id", "action", "tool", "state", "evidence_maturity",
                "duration_ms", "timestamp", "level"):
        assert key in payload


def test_logs_exclude_credentials_paths_and_image_bytes():
    scrubbed = _scrub({
        "authorization": "Bearer abc123",
        "path": "/home/user/photo.jpg",
        "image_bytes": b"\xff\xd8\xff\x00",
        "aws_secret_access_key": "AKIAsecret",
        "run_id": "safe",
    })
    assert scrubbed["authorization"] == "[redacted]"
    assert scrubbed["path"] == "[redacted]"
    # The key name alone is enough to redact it; the value never reaches the
    # bytes branch. Stricter than a length-preserving placeholder would be.
    assert scrubbed["image_bytes"] == "[redacted]"
    assert scrubbed["aws_secret_access_key"] == "[redacted]"
    assert scrubbed["run_id"] == "safe"


def test_raw_bytes_under_a_neutral_key_are_still_omitted():
    scrubbed = _scrub({"payload": b"\xff\xd8\xff\x00"})
    assert "bytes omitted" in scrubbed["payload"]


def test_a_filesystem_path_anywhere_in_a_value_is_dropped():
    assert _scrub("loaded model from /home/x/model.onnx") == "[redacted]"


def test_metrics_are_exposed(client):
    client.post("/inspect",
                files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")})
    snapshot = client.get("/metrics").json()
    assert snapshot["counters"]["inspection_requests_total"] >= 1
    assert "inspection_duration_ms" in snapshot["durations"]


def test_unsafe_inference_counter_stays_zero(client):
    for image in (reference_capture(), defocus(reference_capture(), 21),
                  scene_without_a_subject()):
        client.post("/inspect",
                    files={"image": ("a.jpg", encode(image), "image/jpeg")})
    counters = client.get("/metrics").json()["counters"]
    assert counters.get("unsafe_inference_count", 0) == 0


# --- 15: configuration carries no credential -----------------------------------


def test_the_public_config_summary_contains_no_credential(config):
    payload = json.dumps(config.public_summary()).lower()
    for token in ("secret", "password", "token", "aws_access_key", "credential"):
        assert token not in payload


def test_version_endpoint_is_safe(client):
    body = client.get("/version").json()
    payload = json.dumps(body).lower()
    for token in ("secret", "password", "aws_access_key", "/home/"):
        assert token not in payload
    assert body["opencv_version"].startswith("5.")


def test_the_fallback_default_is_off_in_the_service(config):
    assert config.enable_foreground_fallback is False
    assert config.public_summary()["enable_foreground_fallback"] is False


def test_uploads_are_not_retained_by_default(config):
    assert config.retain_uploads is False


def test_no_food_safety_vocabulary_in_the_substantive_response(client):
    """The findings must never use food-safety vocabulary.

    The two disclaimer fields are excluded, and deliberately so: a claim
    boundary has to be able to name the things it is denying. "This is not a
    food-safety or edibility assessment" is the opposite of a food-safety
    claim, and a check that forbade it would forbid saying so at all. The
    disclaimers are asserted separately below, as negations.
    """
    from competition.models.ontology import PROHIBITED_CLAIM_TERMS

    body = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    ).json()
    disclaimers = {"claim_boundary", "deployment_input_contract"}
    substantive = {k: v for k, v in body.items() if k not in disclaimers}
    payload = json.dumps(substantive).lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in payload, f"prohibited term {term!r} in the API response"


def test_the_disclaimers_are_phrased_as_denials(client):
    body = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    ).json()
    boundary = body["claim_boundary"].lower()
    assert "visible surface condition" in boundary
    assert "not a food-safety" in boundary
    assert "not a food-safety statement" in body["deployment_input_contract"].lower()


def test_condition_evidence_uses_visible_condition_vocabulary(client):
    body = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    ).json()
    evidence = body["condition_evidence"]
    assert evidence is not None
    assert evidence["visible_condition"] in {"fresh", "deteriorated", "uncertain"}


# --- transport domain: JPEG is lossy, PNG is not ------------------------------


def test_png_upload_is_reported_as_lossless(client):
    """PNG preserves the exact array, so API and in-process results agree."""
    body = client.post(
        "/inspect",
        files={"image": ("a.png", encode(reference_capture(), ".png"), "image/png")},
    ).json()
    assert body["lossless_transport"] is True


def test_jpeg_upload_is_reported_as_lossy(client):
    body = client.post(
        "/inspect", files={"image": ("a.jpg", encode(reference_capture()), "image/jpeg")}
    ).json()
    assert body["lossless_transport"] is False


def test_png_transport_matches_the_in_process_result(client, config, model_available):
    """The claim the lossless flag exists to support.

    A PNG upload must reach exactly the same terminal state as inspecting the
    same array directly. If this ever fails, the divergence is in the service
    rather than in the transport.
    """
    if not model_available:
        pytest.skip("model artifact unavailable")
    from competition.agent.orchestrator import load_locked_policies, run_inspection
    from competition.evaluation.phase3_agent_scenarios import load_model

    image = darken(reference_capture(), 0.12)
    model, _ = load_model()
    in_process = run_inspection(image, model, load_locked_policies(), run_id="cmp")

    body = client.post(
        "/inspect", files={"image": ("a.png", encode(image, ".png"), "image/png")}
    ).json()
    assert body["final_status"] == in_process.final_state.value
    assert body["original_image_sha256"] == in_process.original_image_sha256


def test_the_lossless_helper_classifies_formats():
    from competition.service.uploads import is_lossless

    assert is_lossless("image/png")
    assert is_lossless("image/bmp")
    assert not is_lossless("image/jpeg")
