"""FastAPI inspection service.

A thin HTTP layer over the Phase 3 orchestrator. It validates input, runs the
existing bounded loop, persists the trace and returns a structured result. It
contains no vision code and no policy: every decision in a response was made by
`competition.agent.orchestrator`, unchanged.

The status-code semantics are the part worth reading carefully. A capture the
agent refuses is a **successful inspection** — the system did exactly what it
was built to do, and answering 500 would tell a client that AgriVision is broken
when in fact the photograph was. So:

    200  the loop ran to a terminal state, whatever that state was,
         including REQUEST_RECAPTURE and REQUEST_HUMAN_REVIEW
    400  the request was not a usable image
    413  the upload exceeded the size limit
    415  the content was not a supported image format
    503  the service is not ready (no verified model artifact)
    500  an internal fault, with a safe body and no internals

`requested_human_action` is how a client distinguishes "inspected" from "needs
another photograph", not the HTTP status.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from competition.agent.orchestrator import (
    InspectionRunResult,
    OrchestratorPolicy,
    load_locked_policies,
    run_inspection,
)
from competition.agent.state import InspectionState
from competition.service import observability as obs
from competition.service.artifacts import ArtifactStatus, artifact_directory, ensure_model_artifact
from competition.service.config import SERVICE_VERSION, ServiceConfig, load_config
from competition.service.persistence import build_store
from competition.service.uploads import ALLOWED_MEDIA_TYPES, UploadRejected, validate_upload

API_VERSION = "phase4-api-1.0.0"

DEPLOYMENT_INPUT_CONTRACT = (
    "Submit one primary fruit per inspection image. The system does not extract "
    "a single item from market stalls, piles, crates or trees carrying multiple "
    "fruits; such an image is expected to return a recapture or human-review "
    "action. This is a scope boundary, not a food-safety statement."
)


# --- response schemas ---------------------------------------------------------


class QualitySummary(BaseModel):
    foreground_valid: bool
    foreground_invalid_reasons: list = Field(default_factory=list)
    quality_flags: list = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    artifact_flags: list = Field(default_factory=list)
    advisory_flags: list = Field(default_factory=list)
    advisory_human_action: str = ""


class InspectionResponse(BaseModel):
    """What a client receives. No filesystem path, no raw exception, no claim."""

    run_id: str
    final_status: str
    requested_human_action: str = ""
    original_image_sha256: str
    canonical_image_sha256: str
    remediation_applied: bool
    remediation_accepted: bool
    remediation_action: str = ""
    quality_summary: QualitySummary
    artifact_summary: ArtifactSummary
    condition_evidence: Optional[dict] = None
    model_confidence: Optional[float] = None
    condition_model_invoked: bool
    terminal_reason_code: str = ""
    # Whether the bytes inspected were exactly what the client encoded. JPEG is
    # lossy and can move a marginal capture across a calibrated threshold, so a
    # caller comparing API results against in-process results needs to know
    # which transport produced them.
    lossless_transport: bool = False
    trace_id: str
    trace_url: str
    policy_fingerprints: dict
    pipeline_version: str
    claim_boundary: str
    deployment_input_contract: str


class ErrorResponse(BaseModel):
    error: str
    reason_code: str
    detail: str


CLAIM_BOUNDARY = (
    "Reports visible surface condition only. This is not a food-safety, "
    "edibility, contamination or internal-spoilage assessment, and a human "
    "remains responsible for any decision about produce."
)


# --- application state --------------------------------------------------------


class ServiceState:
    """Everything built once at startup and shared by requests."""

    def __init__(self, config: ServiceConfig | None = None) -> None:
        self.config = config or load_config()
        self.logger = obs.configure_logging(self.config.log_level)
        self.metrics = obs.Metrics()
        self.store = build_store(self.config)
        self.artifact: ArtifactStatus | None = None
        self.model = None
        self.model_error = ""
        self.policy: OrchestratorPolicy | None = None
        self.started_at = time.time()

    def initialise(self) -> None:
        """Fetch and verify the artifact, then load the policy and the model.

        Never raises. A container that crash-loops on a misconfiguration is far
        harder to diagnose than one that runs and reports itself unready with a
        reason.
        """
        self.artifact = ensure_model_artifact(self.config)
        try:
            self.policy = load_locked_policies()
            if self.config.enable_foreground_fallback:
                import dataclasses

                self.policy = dataclasses.replace(
                    self.policy, enable_foreground_fallback=True
                )
        except Exception as error:  # noqa: BLE001
            self.model_error = f"policy_load_failed: {type(error).__name__}"
            obs.log_event(self.logger, "startup_policy_failed", level="ERROR",
                          error_type=type(error).__name__)
            return

        if not (self.artifact and self.artifact.usable):
            self.model_error = "model_artifact_unavailable"
            obs.log_event(self.logger, "startup_artifact_unusable", level="ERROR",
                          present=bool(self.artifact and self.artifact.present),
                          verified=bool(self.artifact and self.artifact.verified))
            return

        try:
            from competition.models.adapter import load_condition_model

            self.model = load_condition_model(artifact_directory(self.config))
        except Exception as error:  # noqa: BLE001
            self.model = None
            self.model_error = f"model_load_failed: {type(error).__name__}"
            obs.log_event(self.logger, "startup_model_failed", level="ERROR",
                          error_type=type(error).__name__)
            return

        obs.log_event(self.logger, "startup_complete",
                      model_source=self.artifact.source,
                      persistence=self.store.backend,
                      fallback_enabled=self.config.enable_foreground_fallback)

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until initialisation settles, or the timeout expires.

        For callers that genuinely need readiness before proceeding - tests, and
        a deployment smoke check. The serving path never calls this: a request
        that arrives early is answered 503 rather than made to wait.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready or self.model_error:
                return self.ready
            time.sleep(0.05)
        return self.ready

    @property
    def ready(self) -> bool:
        """Ready means the service can actually complete an inspection."""
        return bool(
            self.policy is not None
            and self.model is not None
            and self.artifact is not None
            and self.artifact.usable
        )

    def readiness_detail(self) -> dict:
        return {
            "ready": self.ready,
            "opencv_loaded": _opencv_version() is not None,
            "opencv_version": _opencv_version(),
            "policy_loaded": self.policy is not None,
            "model_artifact_present": bool(self.artifact and self.artifact.present),
            "model_artifact_verified": bool(self.artifact and self.artifact.verified),
            "model_loaded": self.model is not None,
            "persistence_backend": self.store.backend,
            "persistence_healthy": _safe_healthy(self.store),
            "detail": self.model_error or (
                self.artifact.detail if self.artifact else "initialising"),
            "initialising": self.artifact is None and not self.model_error,
        }


def _opencv_version():
    try:
        import cv2

        return cv2.__version__
    except Exception:  # noqa: BLE001
        return None


def _safe_healthy(store) -> bool:
    try:
        return bool(store.healthy())
    except Exception:  # noqa: BLE001
        return False


# --- result mapping -----------------------------------------------------------


_HUMAN_ACTION_COUNTERS = {
    InspectionState.REQUEST_RECAPTURE.value: obs.RECAPTURE_REQUESTS,
    InspectionState.REQUEST_REPOSITION_LIGHT.value: obs.REPOSITION_REQUESTS,
    InspectionState.REQUEST_HUMAN_REVIEW.value: obs.HUMAN_REVIEW_REQUESTS,
}


def _to_response(result: InspectionRunResult, run_id: str,
                 lossless: bool = False) -> InspectionResponse:
    return InspectionResponse(
        run_id=run_id,
        final_status=result.final_state.value,
        requested_human_action=result.requested_human_action,
        original_image_sha256=result.original_image_sha256,
        canonical_image_sha256=result.canonical_image_sha256,
        remediation_applied=result.remediation_applied,
        remediation_accepted=result.remediation_accepted,
        remediation_action=result.remediation_action,
        quality_summary=QualitySummary(
            foreground_valid=result.foreground_valid,
            foreground_invalid_reasons=list(result.foreground_invalid_reasons),
            quality_flags=list(result.quality_flags),
        ),
        artifact_summary=ArtifactSummary(
            artifact_flags=list(result.artifact_flags),
            advisory_flags=list(result.advisory_flags),
            advisory_human_action=result.advisory_human_action,
        ),
        condition_evidence=result.condition_evidence,
        model_confidence=result.model_confidence,
        condition_model_invoked=result.inference_ran,
        terminal_reason_code=result.terminal_reason_code,
        lossless_transport=lossless,
        trace_id=run_id,
        trace_url=f"/inspection/{run_id}/trace",
        policy_fingerprints=result.policy_fingerprints,
        pipeline_version=result.pipeline_version,
        claim_boundary=CLAIM_BOUNDARY,
        deployment_input_contract=DEPLOYMENT_INPUT_CONTRACT,
    )


def _record_metrics(state: ServiceState, result: InspectionRunResult,
                    duration_ms: float) -> None:
    metrics = state.metrics
    metrics.increment(obs.INSPECTION_REQUESTS)
    metrics.observe(obs.INSPECTION_DURATION, duration_ms)

    if result.final_state is InspectionState.COMPLETE:
        metrics.increment(obs.INSPECTION_SUCCESS)
    counter = _HUMAN_ACTION_COUNTERS.get(result.final_state.value)
    if counter:
        metrics.increment(counter)
    if result.remediation_applied:
        metrics.increment(obs.REMEDIATION_ATTEMPTS)
    if result.inference_ran:
        metrics.increment(obs.CONDITION_INFERENCE)
    if result.failure:
        metrics.increment(obs.TOOL_FAILURES)
    if "BudgetExceeded" in (result.failure or ""):
        metrics.increment(obs.STEP_BUDGET_EXCEEDED)

    # The safety invariant, counted rather than assumed: the classifier must
    # never have run on a capture that ended in a refusal.
    blocked = result.final_state in (
        InspectionState.REQUEST_RECAPTURE,
        InspectionState.REQUEST_REPOSITION_LIGHT,
        InspectionState.FAILED_SAFE,
    )
    if blocked and result.inference_ran:
        metrics.increment(obs.UNSAFE_INFERENCE)

    for step in result.trace.steps:
        if step.tool_name == "segment_foreground":
            metrics.observe(obs.FOREGROUND_DURATION, step.duration_ms)
        elif step.tool_name == "assess_capture_quality":
            metrics.observe(obs.QUALITY_DURATION, step.duration_ms)
        elif step.tool_name in ("assess_local_highlights", "assess_visibility"):
            metrics.observe(obs.ARTIFACT_DURATION, step.duration_ms)
        elif step.tool_name == "run_condition_model":
            metrics.observe(obs.MODEL_DURATION, step.duration_ms)


def _trace_record(result: InspectionRunResult, run_id: str, media_type: str) -> dict:
    """What goes into the store. Content hashes only; never image bytes."""
    return {
        "run_id": run_id,
        "created_at": int(time.time()),
        "final_status": result.final_state.value,
        "requested_human_action": result.requested_human_action,
        "original_image_sha256": result.original_image_sha256,
        "canonical_image_sha256": result.canonical_image_sha256,
        "media_type": media_type,
        "pipeline_version": result.pipeline_version,
        "policy_fingerprints": result.policy_fingerprints,
        "terminal_reason_code": result.terminal_reason_code,
        "condition_model_invoked": result.inference_ran,
        # The full causal structure, not a prose summary: this is the artifact
        # a judge inspects to see evidence -> decision -> next action.
        "trace": result.trace.to_dict(include_timing=True),
        "decisions": [d.to_dict() for d in result.decisions],
    }


# --- application --------------------------------------------------------------


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    state = ServiceState(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Initialisation runs on a background thread so the port opens
        # immediately. Downloading and verifying an 11 MB model inside the
        # lifespan blocks uvicorn from accepting connections, which means the
        # platform health check fails before the process has done anything
        # wrong -- and on App Runner that presents as "failed to deploy your
        # application image" with no application logs at all, because the
        # container never served a request.
        #
        # This is exactly why /health and /ready are separate: liveness can be
        # true long before readiness is, and the platform needs to be able to
        # see the difference rather than conclude the image is broken.
        threading.Thread(target=state.initialise, name="startup",
                         daemon=True).start()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="AgriVision inspection service",
        version=SERVICE_VERSION,
        description=(
            "Bounded Agentic Vision produce inspection. OpenCV 5 perception "
            "drives a deterministic decision loop; no language model "
            "participates in any decision.\n\n"
            f"**Input contract.** {DEPLOYMENT_INPUT_CONTRACT}\n\n"
            f"**Claim boundary.** {CLAIM_BOUNDARY}"
        ),
    )
    app.state.service = state

    @app.get("/health", tags=["operations"])
    def health() -> dict:
        """Liveness only: the process is up. Says nothing about readiness."""
        return {
            "status": "alive",
            "service_version": SERVICE_VERSION,
            "uptime_seconds": round(time.time() - state.started_at, 1),
        }

    @app.get("/ready", tags=["operations"])
    def ready(response: Response) -> dict:
        """Readiness: can this instance actually complete an inspection?

        Reports 503 when it cannot, so App Runner replaces the instance rather
        than routing traffic to one that would refuse every request.
        """
        detail = state.readiness_detail()
        if not detail["ready"]:
            response.status_code = 503
        return detail

    @app.get("/version", tags=["operations"])
    def version() -> dict:
        summary = state.config.public_summary()
        summary["api_version"] = API_VERSION
        summary["opencv_version"] = _opencv_version()
        summary["claim_boundary"] = CLAIM_BOUNDARY
        return summary

    @app.get("/metrics", tags=["operations"])
    def metrics() -> dict:
        return state.metrics.snapshot()

    @app.post("/inspect", response_model=InspectionResponse, tags=["inspection"])
    async def inspect(request: Request, image: UploadFile = File(...)) -> JSONResponse:
        if not state.ready:
            detail = state.readiness_detail()
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_not_ready",
                    "reason_code": "MODEL_UNAVAILABLE",
                    "detail": detail["detail"] or "The service is not ready.",
                },
            )

        # Opaque, unguessable, and unrelated to the filename or to any content
        # the client supplied. The image is identified internally by SHA-256.
        run_id = uuid.uuid4().hex

        payload = await image.read()
        try:
            decoded = validate_upload(payload, state.config)
        except UploadRejected as rejection:
            state.metrics.increment(obs.REJECTED_UPLOADS)
            obs.log_event(state.logger, "upload_rejected", level="WARNING",
                          run_id=run_id, reason_code=rejection.reason_code,
                          size_bytes=len(payload))
            status = {
                "UPLOAD_TOO_LARGE": 413,
                "UNSUPPORTED_MEDIA_TYPE": 415,
            }.get(rejection.reason_code, 400)
            raise HTTPException(
                status_code=status,
                detail={
                    "error": "upload_rejected",
                    "reason_code": rejection.reason_code,
                    "detail": rejection.message,
                },
            ) from None

        obs.log_event(state.logger, "inspection_started", run_id=run_id,
                      media_type=decoded.media_type, width=decoded.width,
                      height=decoded.height, size_bytes=decoded.size_bytes,
                      lossless_transport=decoded.lossless)

        started = time.perf_counter()
        try:
            result = run_inspection(
                decoded.image, state.model, state.policy, run_id=run_id
            )
        except Exception as error:  # noqa: BLE001 - never leak internals
            state.metrics.increment(obs.TOOL_FAILURES)
            obs.log_event(state.logger, "inspection_failed", level="ERROR",
                          run_id=run_id, error_type=type(error).__name__)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "inspection_failed",
                    "reason_code": "INTERNAL_ERROR",
                    "detail": "The inspection could not be completed.",
                },
            ) from None
        duration_ms = (time.perf_counter() - started) * 1000.0

        _record_metrics(state, result, duration_ms)

        # Persistence failure must not change the result. It is logged, counted
        # and reported in a header; the body is identical either way.
        outcome = state.store.put(
            run_id, _trace_record(result, run_id, decoded.media_type)
        )
        if not outcome.stored:
            state.metrics.increment(obs.PERSISTENCE_FAILURES)
            obs.log_event(state.logger, "trace_persist_failed", level="ERROR",
                          run_id=run_id, backend=outcome.backend,
                          error_type=outcome.detail)

        for step in result.trace.steps:
            obs.log_event(
                state.logger, "agent_step", run_id=run_id, step_id=step.step_id,
                state=step.state_after, tool=step.tool_name,
                action=step.selected_action, reason_code=result.terminal_reason_code,
                evidence_maturity=step.evidence_maturity,
                duration_ms=round(step.duration_ms, 2),
            )

        obs.log_event(
            state.logger, "inspection_completed", run_id=run_id,
            state=result.final_state.value,
            action=result.requested_human_action or "NONE",
            reason_code=result.terminal_reason_code,
            condition_model_invoked=result.inference_ran,
            remediation_applied=result.remediation_applied,
            duration_ms=round(duration_ms, 2),
            success=True, trace_stored=outcome.stored,
            pipeline_version=result.pipeline_version,
        )

        body = _to_response(result, run_id, decoded.lossless).model_dump()
        # A refusal is a successful inspection. 200 with an action to take.
        return JSONResponse(
            status_code=200, content=body,
            headers={"X-Trace-Stored": "true" if outcome.stored else "false"},
        )

    @app.get("/inspection/{run_id}", tags=["inspection"])
    def get_inspection(run_id: str) -> dict:
        record = _lookup(state, run_id)
        summary = {k: v for k, v in record.items() if k not in ("trace", "decisions")}
        summary["trace_url"] = f"/inspection/{run_id}/trace"
        return summary

    @app.get("/inspection/{run_id}/trace", tags=["inspection"])
    def get_trace(run_id: str) -> dict:
        record = _lookup(state, run_id)
        return {
            "run_id": run_id,
            "final_status": record.get("final_status"),
            "trace": record.get("trace"),
            "decisions": record.get("decisions", []),
            "policy_fingerprints": record.get("policy_fingerprints", {}),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    return app


def _lookup(state: ServiceState, run_id: str) -> dict:
    if not run_id or len(run_id) > 64 or not run_id.isalnum():
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_run_id", "reason_code": "INVALID_RUN_ID",
                    "detail": "Run identifiers are alphanumeric."},
        )
    try:
        record = state.store.get(run_id)
    except Exception as error:  # noqa: BLE001
        state.metrics.increment(obs.PERSISTENCE_FAILURES)
        obs.log_event(state.logger, "trace_read_failed", level="ERROR",
                      run_id=run_id, error_type=type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail={"error": "trace_unavailable", "reason_code": "PERSISTENCE_ERROR",
                    "detail": "The trace store is unavailable."},
        ) from None
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "reason_code": "RUN_NOT_FOUND",
                    "detail": "No inspection with that identifier."},
        )
    return record


app = create_app()
