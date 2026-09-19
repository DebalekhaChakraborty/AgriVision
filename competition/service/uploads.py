"""Upload validation.

A filename extension is a claim by the client, not a fact about the bytes, so
nothing here trusts one. Validation is done on decoded content: the magic bytes
say what the file is, and OpenCV decoding it successfully says it is actually an
image rather than a truncated or crafted file that merely begins like one.

Limits exist to bound work, not to be strict for its own sake. An unbounded
upload is a memory exhaustion vector, and a 40000x40000 PNG that decompresses to
several gigabytes is a denial of service dressed as a photograph.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from competition.service.config import ServiceConfig

# Magic-byte signatures for the formats the service accepts. Checked against the
# bytes received, never against the declared content type or the filename.
_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)
_WEBP_PREFIX = b"RIFF"
_WEBP_TAG = b"WEBP"

ALLOWED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
)


# PNG is lossless and JPEG is not. That distinction is not cosmetic here: a
# marginal capture can cross a calibrated decision boundary purely because of
# JPEG quantisation. Measured in Phase 4: darken(reference, 0.12) completes
# in-process, and after a JPEG round trip has gamma applied and accepted and is
# then refused by the severe-contrast check, with pre-remediation luminance
# differing only in the fourth decimal (0.0480 vs 0.0485).
#
# The thresholds were NOT loosened to hide this. The agent is responding
# correctly to the pixels it was actually given, and JPEG results are transport
# domain behaviour rather than a defect. Where a caller needs the bytes the
# service sees to match an in-process array exactly -- a deterministic test, a
# reproducible demonstration -- PNG is the format to send.
LOSSLESS_MEDIA_TYPES = frozenset({"image/png", "image/bmp", "image/tiff"})


def is_lossless(media_type: str) -> bool:
    """Whether this format preserves the exact pixels the client encoded."""
    return media_type in LOSSLESS_MEDIA_TYPES


class UploadRejected(ValueError):
    """Raised for input that is not an acceptable image. Always a 4xx."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


@dataclass(frozen=True)
class DecodedUpload:
    """A validated image, ready for the orchestrator."""

    image: np.ndarray
    media_type: str
    size_bytes: int
    width: int
    height: int

    @property
    def lossless(self) -> bool:
        """True when the decoded array is exactly what the client encoded."""
        return is_lossless(self.media_type)


def sniff_media_type(payload: bytes) -> str:
    """Identify the format from its leading bytes. Returns "" if unrecognised."""
    for signature, media_type in _SIGNATURES:
        if payload.startswith(signature):
            return media_type
    if payload[:4] == _WEBP_PREFIX and payload[8:12] == _WEBP_TAG:
        return "image/webp"
    return ""


def validate_upload(payload: bytes, config: ServiceConfig) -> DecodedUpload:
    """Validate and decode, or raise `UploadRejected` with a machine-readable code.

    Order matters. Size is checked before decoding so a hostile payload is never
    handed to a decoder, and dimensions are checked after decoding because a
    header can lie about them.
    """
    if not payload:
        raise UploadRejected("EMPTY_UPLOAD", "The request contained no image data.")

    if len(payload) > config.max_upload_bytes:
        raise UploadRejected(
            "UPLOAD_TOO_LARGE",
            f"Image exceeds the {config.max_upload_bytes} byte limit.",
        )

    media_type = sniff_media_type(payload)
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise UploadRejected(
            "UNSUPPORTED_MEDIA_TYPE",
            "Content is not a recognised image. Accepted formats: "
            + ", ".join(sorted(ALLOWED_MEDIA_TYPES)) + ".",
        )

    buffer = np.frombuffer(payload, dtype=np.uint8)
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise UploadRejected(
            "UNDECODABLE_IMAGE", "The image could not be decoded."
        ) from error

    if image is None or image.size == 0:
        raise UploadRejected(
            "UNDECODABLE_IMAGE",
            "The image could not be decoded; it may be truncated or corrupt.",
        )

    height, width = image.shape[:2]
    if min(height, width) < config.min_image_dimension:
        raise UploadRejected(
            "IMAGE_TOO_SMALL",
            f"Both dimensions must be at least {config.min_image_dimension} px.",
        )
    if max(height, width) > config.max_image_dimension:
        raise UploadRejected(
            "IMAGE_TOO_LARGE",
            f"Neither dimension may exceed {config.max_image_dimension} px.",
        )
    if height * width > config.max_image_pixels:
        raise UploadRejected(
            "IMAGE_TOO_MANY_PIXELS",
            f"Image exceeds {config.max_image_pixels} pixels.",
        )

    # cv2.imdecode with IMREAD_COLOR always yields 3-channel BGR, which is what
    # the perception layer's contract requires. Asserted rather than assumed
    # because the whole pipeline downstream depends on it.
    if image.ndim != 3 or image.shape[2] != 3:
        raise UploadRejected(
            "UNDECODABLE_IMAGE", "The image did not decode to three channels."
        )

    return DecodedUpload(
        image=image, media_type=media_type, size_bytes=len(payload),
        width=int(width), height=int(height),
    )
