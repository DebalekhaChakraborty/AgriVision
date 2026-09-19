"""Phase 5: the judge demo page, its claims, and what it must never show."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
from competition.service.api import create_app
from competition.service.artifacts import ensure_model_artifact
from competition.service.config import ServiceConfig
from competition.service.counterfactual import (
    CONTROLLED_DEMONSTRATION_NOTICE,
    VARIANTS,
)

ARTIFACT_DIR = "competition/models/artifacts/mobilenetv3_large_v2exp004"
STATIC = Path("competition/service/static")


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


@pytest.fixture(scope="module")
def config() -> ServiceConfig:
    return ServiceConfig(model_dir=ARTIFACT_DIR)


@pytest.fixture(scope="module")
def client(config):
    if not ensure_model_artifact(config).usable:
        pytest.skip("model artifact unavailable in this checkout")
    app = create_app(config)
    with TestClient(app) as test_client:
        assert app.state.service.wait_until_ready(60)
        yield test_client


@pytest.fixture(scope="module")
def page() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


# --- 1: the page serves --------------------------------------------------------


def test_home_page_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AgriVision" in response.text


def test_static_assets_serve(client):
    for asset in ("/static/app.css", "/static/app.js"):
        assert client.get(asset).status_code == 200


def test_the_page_is_served_same_origin_as_the_api(client):
    """No separate SPA host, so no CORS surface and one certificate."""
    assert client.get("/").status_code == 200
    assert client.get("/ready").status_code == 200


# --- 2/3: the page drives the live API ----------------------------------------


def test_the_page_posts_to_the_live_inspect_endpoint(script):
    assert '"/inspect"' in script
    assert '"/counterfactual"' in script
    assert "/ready" in script


def test_result_status_is_rendered_from_the_response(script):
    assert "final_status" in script
    assert "COMPLETE" in script and "REQUEST_RECAPTURE" in script


# --- 4/5: conditional rendering ------------------------------------------------


def test_condition_result_renders_only_when_inference_ran(script):
    assert re.search(r"if \(r\.condition_evidence\)", script)
    assert "modelResult" in script


def test_human_action_copy_exists_for_every_human_terminal(script):
    for action in ("REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT",
                   "REQUEST_HUMAN_REVIEW"):
        assert action in script


def test_failure_copy_is_guidance_not_an_error(script):
    assert "Please retake the photo" in script
    assert "reposition the light" in script.lower()
    assert "could not establish a trustworthy single-subject foreground" in script.lower()


def test_no_stack_trace_is_ever_rendered(script):
    assert "traceback" not in script.lower()
    assert "stack" not in script.lower()


# --- 6/7: trace and model-skipped state ---------------------------------------


def test_inspect_returns_the_trace_the_page_renders(client):
    body = client.post(
        "/inspect", files={"image": ("a.png", encode(reference_capture()), "image/png")}
    ).json()
    assert body["trace_steps"]
    assert [s["step_id"] for s in body["trace_steps"]] == list(
        range(1, len(body["trace_steps"]) + 1))
    assert body["state_path"][0] == "RECEIVED"


def test_the_trace_preserves_causal_order_and_fields(client):
    body = client.post(
        "/inspect",
        files={"image": ("b.png", encode(defocus(reference_capture(), 21)), "image/png")},
    ).json()
    steps = body["trace_steps"]
    for earlier, later in zip(steps, steps[1:]):
        assert later["state_before"] == earlier["state_after"]
    quality = next(s for s in steps if s["tool_name"] == "assess_capture_quality")
    assert "high_frequency_ratio" in quality["evidence_summary"]
    assert "focus_floor" in quality["evidence_summary"]


def test_model_skipped_is_visible_in_the_response(client):
    body = client.post(
        "/inspect",
        files={"image": ("b.png", encode(defocus(reference_capture(), 21)), "image/png")},
    ).json()
    assert body["condition_model_invoked"] is False
    assert body["condition_evidence"] is None


def test_the_page_states_when_the_model_was_not_invoked(script):
    assert "NOT</strong> invoked" in script or "not invoked" in script


def test_raw_trace_is_behind_an_expandable_section(page):
    assert "<details" in page
    assert "Technical trace" in page
    assert 'id="rawTrace"' in page


# --- 8/9: evidence maturity ----------------------------------------------------


def test_maturity_badges_exist_for_each_level(page):
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    for level in ("CALIBRATED", "PROVISIONAL", "ADVISORY", "UNQUALIFIED"):
        assert f".badge.{level}" in css


def test_glare_is_presented_as_advisory_not_calibrated(script):
    """The distinction Phase 2d measured must survive into the UI."""
    assert "advisory_human_action" in script
    assert 'badge ADVISORY' in script
    assert "did <em>not</em> block" in script


def test_advisory_glare_does_not_block_and_the_page_says_so(client):
    from competition.vision.degradation import add_glare_at

    glared, _ = add_glare_at(reference_capture(), intensity=0.5,
                             centre=(170, 170), radius=45, falloff=0.45)
    body = client.post(
        "/inspect", files={"image": ("g.png", encode(glared), "image/png")}
    ).json()
    if body["artifact_summary"]["advisory_human_action"]:
        assert body["requested_human_action"] == ""
        assert body["final_status"] == "COMPLETE"


def test_fallback_masks_are_shown_as_provisional(script):
    assert "FALLBACK" in script
    assert 'badge PROVISIONAL' in script


# --- 10/11: responsible use ----------------------------------------------------


def test_responsible_use_statement_is_present(page):
    lowered = page.lower()
    assert "does not detect pathogens" in lowered
    assert "internal spoilage" in lowered
    assert "safe to eat" in lowered


def test_the_hero_does_not_claim_food_safety(page):
    assert "determines whether food is safe to eat" not in page.lower()


def test_the_page_uses_visible_condition_vocabulary(page, script):
    assert "visible produce condition" in page.lower()
    assert "visible-condition model" in script.lower()


def test_no_food_safety_claim_outside_the_disclaimer(page):
    """The disclaimer may name what it denies; nothing else may."""
    from competition.models.ontology import PROHIBITED_CLAIM_TERMS

    body = re.sub(r'<section class="card responsible".*?</section>', "", page,
                  flags=re.S)
    lowered = body.lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in lowered, f"prohibited term {term!r} outside the disclaimer"


def test_the_input_contract_is_stated_on_the_page(page):
    assert "one primary fruit per image" in page.lower()


# --- 12/13: the counterfactual -------------------------------------------------


def test_counterfactual_runs_the_live_agent_on_every_variant(client):
    body = client.post(
        "/counterfactual",
        files={"image": ("a.png", encode(reference_capture()), "image/png")},
    ).json()
    assert body["variant_count"] == len(VARIANTS)
    for variant in body["variants"]:
        assert variant["state_path"], "each variant is a real run with a real path"
        assert variant["final_status"]


def test_the_counterfactual_shows_evidence_changing_the_action(client):
    """The central competition demonstration."""
    body = client.post(
        "/counterfactual",
        files={"image": ("a.png", encode(reference_capture()), "image/png")},
    ).json()
    assert body["evidence_changes_action"] is True
    assert len(body["distinct_first_actions"]) >= 3


def test_each_variant_takes_the_action_its_evidence_implies(client):
    body = client.post(
        "/counterfactual",
        files={"image": ("a.png", encode(reference_capture()), "image/png")},
    ).json()
    actions = {v["key"]: v["first_action"] for v in body["variants"]}
    invoked = {v["key"]: v["condition_model_invoked"] for v in body["variants"]}
    assert actions["REFERENCE"] == "NONE"
    assert actions["UNDEREXPOSED"] == "APPLY_GAMMA"
    assert actions["SEVERE_BLUR"] == "REQUEST_RECAPTURE"
    assert invoked["SEVERE_BLUR"] is False
    assert invoked["REFERENCE"] is True


def test_the_counterfactual_is_labelled_a_controlled_demonstration(client, page, script):
    body = client.post(
        "/counterfactual",
        files={"image": ("a.png", encode(reference_capture()), "image/png")},
    ).json()
    assert body["notice"] == CONTROLLED_DEMONSTRATION_NOTICE
    assert "not a real-world accuracy benchmark" in body["notice"].lower()
    assert "cfNotice" in page and "cfNotice" in script


def test_variants_are_derived_from_the_upload_and_not_persisted(client):
    """Avoids a licensing problem: no bundled demo photograph exists."""
    body = client.post(
        "/counterfactual",
        files={"image": ("a.png", encode(reference_capture()), "image/png")},
    ).json()
    assert "not persisted" in body["source"]


def test_counterfactual_rejects_a_non_image(client):
    response = client.post(
        "/counterfactual", files={"image": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 415


# --- 15/16: degraded states ----------------------------------------------------


def test_the_page_handles_a_service_that_is_not_ready(config, script):
    broken = ServiceConfig(model_dir="/nonexistent/path")
    app = create_app(broken)
    with TestClient(app) as probe:
        app.state.service.wait_until_ready(20)
        assert probe.get("/").status_code == 200        # page still serves
        assert probe.get("/ready").status_code == 503
    assert "Service not ready" in script


def test_a_refused_upload_is_reported_without_internals(client):
    response = client.post(
        "/inspect", files={"image": ("x.png", b"garbage", "image/png")})
    body = response.json()
    assert response.status_code == 415
    assert set(body["detail"]) == {"error", "reason_code", "detail"}
    assert "Traceback" not in json.dumps(body)


def test_the_page_handles_an_unreadable_api_response(script):
    assert "could not read" in script


# --- 17: nothing internal leaks ------------------------------------------------


def test_no_aws_or_filesystem_identifier_on_the_page(page, script):
    for marker in ("s3://", "amazonaws.com", "/home/", "/opt/", "arn:aws:",
                   "dynamodb"):
        assert marker not in page, f"{marker!r} leaked into the page"
        assert marker not in script, f"{marker!r} leaked into the script"


def test_no_aws_account_number_on_the_page(page, script):
    """Matched by shape, so the account id never has to live in this repository."""
    account_like = re.compile(r"(?<!\d)\d{12}(?!\d)")
    for name, text in (("page", page), ("script", script)):
        found = account_like.search(text)
        assert found is None, f"12-digit account-like number in the {name}"


def test_the_inspection_response_carries_no_path_or_bucket(client):
    body = client.post(
        "/inspect", files={"image": ("secret_name.png", encode(reference_capture()),
                                     "image/png")}
    ).json()
    payload = json.dumps(body)
    for marker in ("/home/", "s3://", "arn:aws:", "secret_name", ".png"):
        assert marker not in payload


# --- 22: no fabricated results -------------------------------------------------


def test_the_page_contains_no_hard_coded_inspection_result(script):
    """Everything rendered must come from a live response."""
    assert "renderResult(lastRun)" in script
    for fake in ("fresh_apple", "0.9234", "RECORDED EXAMPLE"):
        assert fake not in script


def test_every_rendered_field_comes_from_the_response_object(script):
    for field in ("r.final_status", "r.condition_model_invoked", "r.run_id",
                  "r.trace_steps", "r.policy_fingerprints"):
        assert field in script


# --- 19/21: path display and accessibility -------------------------------------


def test_only_the_traversed_states_are_shown(script):
    assert "r.state_path" in script
    assert "18" not in script.split("renderPath")[1][:400]


def test_keyboard_file_selection_is_supported(page, script):
    assert 'tabindex="0"' in page
    assert 'keydown' in script and '"Enter"' in script


def test_the_page_has_accessible_structure(page):
    assert 'lang="en"' in page
    assert 'aria-live' in page
    assert 'class="skip"' in page
    assert page.count("aria-labelledby") >= 5


def test_a_reset_action_exists(page, script):
    assert 'id="reset"' in page
    assert 'New inspection' in page


def test_reduced_motion_is_respected():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css


def test_the_layout_is_responsive():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "@media (max-width" in css
    assert "minmax(" in css
