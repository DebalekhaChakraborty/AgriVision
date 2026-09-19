"""Model artifact acquisition with fail-closed verification.

The ONNX export is not in Git, deliberately: it derives from a research
checkpoint whose source dataset licence is recorded as Unknown with no
redistribution grant. Committing it to simplify deployment would trade a
licensing position for convenience.

So the container fetches it at startup and verifies it by content before
`cv2.dnn` ever sees it:

    S3 object  ->  local file  ->  SHA-256 check  ->  cv2.dnn

Three properties matter more than the mechanism.

**The expected hash is configuration, not discovery.** It is compiled into the
image and may be overridden by an operator, but it is never read from the same
place as the artifact — a checksum shipped alongside the file it checks proves
only that the file arrived intact.

**A mismatch is fatal.** Not a warning, not a degraded mode: the service stays
unready and says why. A classifier that runs on an unverified graph is worse
than one that refuses to start, because its answers look exactly like correct
ones.

**No arbitrary URL.** The artifact comes from a configured S3 bucket the task
role can read, or from a local path for development. There is no code path that
downloads a model from a URL supplied at request time.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from competition.service.config import ServiceConfig

CHUNK = 1024 * 1024


class ArtifactError(RuntimeError):
    """Raised when the model artifact is missing, unreadable or unverified."""


@dataclass(frozen=True)
class ArtifactStatus:
    """What the service knows about its model artifact. Carries no path."""

    present: bool
    verified: bool
    source: str
    expected_sha256: str
    observed_sha256: str = ""
    size_bytes: int = 0
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.present and self.verified

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_from_s3(config: ServiceConfig, destination: Path) -> str:
    """Fetch the artifact using the task role. Never takes a caller-supplied URL.

    Both objects are required. `load_condition_model` reads `manifest.json` to
    check the preprocessing contract against the source checkpoint, and refuses
    to load without it -- which is the correct behaviour, since a graph without
    its preprocessing description can be fed inputs it was never trained on.
    Downloading only the graph produced a service that verified its checksum,
    reported the artifact present and verified, and still could not load it.
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    destination.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=config.aws_region)
    try:
        client.download_file(config.model_bucket, config.model_key, str(destination))
        client.download_file(
            config.model_bucket, config.manifest_key,
            str(destination.parent / "manifest.json"),
        )
    except (BotoCoreError, ClientError) as error:
        # The bucket and key are operator configuration, not user input, so
        # naming them here helps an operator and leaks nothing to a client:
        # this string never reaches an HTTP response body.
        raise ArtifactError(
            f"could not fetch s3://{config.model_bucket}/{config.model_key}: "
            f"{type(error).__name__}"
        ) from error
    return f"s3://{config.model_bucket}/{config.model_key}"


def ensure_model_artifact(config: ServiceConfig) -> ArtifactStatus:
    """Make the artifact locally available and verify it. Never raises.

    Returns a status rather than throwing so the caller can decide: readiness
    reports it, and the inspection path refuses to run without it. A failure
    here must not take the process down, because a live-but-unready container
    is diagnosable and a crash-looping one is not.
    """
    destination = Path(config.model_dir) / "model.onnx"
    source = "local"

    if not destination.is_file() and config.uses_s3_model:
        try:
            source = _download_from_s3(config, destination)
        except ArtifactError as error:
            return ArtifactStatus(
                present=False, verified=False, source="s3",
                expected_sha256=config.expected_model_sha256, detail=str(error),
            )

    if not destination.is_file():
        return ArtifactStatus(
            present=False, verified=False, source=source,
            expected_sha256=config.expected_model_sha256,
            detail=(
                "model artifact not present and no MODEL_BUCKET configured; "
                "the service cannot become ready"
            ),
        )

    observed = sha256_file(destination)
    size = destination.stat().st_size
    if observed != config.expected_model_sha256:
        # Do not delete the file: an operator needs to inspect what arrived.
        return ArtifactStatus(
            present=True, verified=False, source=source,
            expected_sha256=config.expected_model_sha256,
            observed_sha256=observed, size_bytes=size,
            detail=(
                "model artifact checksum mismatch; refusing to load. The "
                "service will not report ready."
            ),
        )

    return ArtifactStatus(
        present=True, verified=True, source=source,
        expected_sha256=config.expected_model_sha256,
        observed_sha256=observed, size_bytes=size,
        detail="verified",
    )


def artifact_directory(config: ServiceConfig) -> Path:
    """Directory `load_condition_model` is pointed at once verification passes."""
    return Path(config.model_dir)
