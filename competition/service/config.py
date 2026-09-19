"""Runtime configuration for the inspection service.

Everything here comes from environment variables with safe defaults, so the
container image carries no secrets and no deployment-specific values. Nothing in
this module reads an AWS credential: the container is given a task role, and
boto3 resolves it from the instance metadata service.

The model artifact is identified by its **expected SHA-256**, recorded here and
checked after download. A mismatch is fatal at startup rather than a warning,
because a service that will happily run an unverified classifier is worse than
one that refuses to start.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

SERVICE_VERSION = "phase4-service-1.0.0"

# The Phase 2 export, identified by content. This is the only model the service
# will load; anything else fails the check and the service stays unready.
EXPECTED_MODEL_SHA256 = (
    "77d8614671873cb41ebfd725f0a4b468beca9c30cfe6a3361217343917324d4b"
)
EXPECTED_MODEL_BYTES = 11906184
MODEL_ID = "mobilenet_v3_large-v2expV2-P2-MOBILENETV3-LARGE-004"


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ServiceConfig:
    """Resolved runtime configuration. Contains no secret and no credential."""

    # --- model artifact ------------------------------------------------------
    model_bucket: str = field(default_factory=lambda: os.environ.get("MODEL_BUCKET", ""))
    model_key: str = field(default_factory=lambda: os.environ.get(
        "MODEL_KEY", "models/mobilenetv3_large_v2exp004/model.onnx"))
    manifest_key: str = field(default_factory=lambda: os.environ.get(
        "MANIFEST_KEY", "models/mobilenetv3_large_v2exp004/manifest.json"))
    model_dir: str = field(default_factory=lambda: os.environ.get(
        "MODEL_DIR", "/opt/agrivision/model"))
    expected_model_sha256: str = field(default_factory=lambda: os.environ.get(
        "EXPECTED_MODEL_SHA256", EXPECTED_MODEL_SHA256))

    # --- persistence ---------------------------------------------------------
    trace_table: str = field(default_factory=lambda: os.environ.get("TRACE_TABLE", ""))
    evidence_bucket: str = field(default_factory=lambda: os.environ.get(
        "EVIDENCE_BUCKET", ""))
    trace_ttl_days: int = field(default_factory=lambda: _int("TRACE_TTL_DAYS", 14))

    # --- upload limits -------------------------------------------------------
    # 12 MB accommodates a modern phone capture with headroom. Enforced on the
    # byte count actually read, never on a Content-Length header a client sets.
    max_upload_bytes: int = field(default_factory=lambda: _int(
        "MAX_UPLOAD_BYTES", 12 * 1024 * 1024))
    max_image_pixels: int = field(default_factory=lambda: _int(
        "MAX_IMAGE_PIXELS", 50_000_000))
    min_image_dimension: int = field(default_factory=lambda: _int(
        "MIN_IMAGE_DIMENSION", 32))
    max_image_dimension: int = field(default_factory=lambda: _int(
        "MAX_IMAGE_DIMENSION", 12000))

    # --- input lifecycle -----------------------------------------------------
    # Default OFF. An inspection does not need the photograph after it has been
    # measured, and retaining user uploads by accident is the kind of default
    # that is only noticed later. Turning it on is deliberate and is recorded in
    # /version so an operator can see what the running service does.
    retain_uploads: bool = field(default_factory=lambda: _flag("RETAIN_UPLOADS", False))
    upload_retention_days: int = field(default_factory=lambda: _int(
        "UPLOAD_RETENTION_DAYS", 7))

    # --- agent policy --------------------------------------------------------
    # Phase 3b measured the fallback as reliable on single-subject captures and
    # wrong on 18 of 19 multi-subject scenes. It stays off unless an operator
    # explicitly enables it, and /version reports which way it is set.
    enable_foreground_fallback: bool = field(default_factory=lambda: _flag(
        "ENABLE_FOREGROUND_FALLBACK", False))

    aws_region: str = field(default_factory=lambda: os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    @property
    def uses_s3_model(self) -> bool:
        return bool(self.model_bucket)

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.trace_table)

    def public_summary(self) -> dict:
        """Safe to return from /version. No bucket contents, no credentials.

        Bucket and table *names* are included because an operator needs to know
        which deployment they are talking to, and a name is not a grant: access
        is controlled by the task role, not by obscurity.
        """
        return {
            "service_version": SERVICE_VERSION,
            "model_id": MODEL_ID,
            "expected_model_sha256": self.expected_model_sha256,
            "region": self.aws_region,
            "model_source": "s3" if self.uses_s3_model else "local",
            "persistence": "dynamodb" if self.persistence_enabled else "in_memory",
            "retain_uploads": self.retain_uploads,
            "upload_retention_days": (
                self.upload_retention_days if self.retain_uploads else None),
            "trace_ttl_days": self.trace_ttl_days,
            "enable_foreground_fallback": self.enable_foreground_fallback,
            "max_upload_bytes": self.max_upload_bytes,
            "deployment_input_contract": (
                "One primary produce item per inspection capture. Market stalls, "
                "piles, crates and trees carrying multiple fruits are outside "
                "scope and are expected to return a recapture or human-review "
                "action. This is a scope boundary, not a food-safety statement."
            ),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def load_config() -> ServiceConfig:
    return ServiceConfig()
