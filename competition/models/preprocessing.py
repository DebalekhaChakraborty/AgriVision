"""Preprocessing contract for condition inference.

Preprocessing is part of the model contract, not an implementation detail. A
silent mismatch here degrades accuracy without raising anything, and the most
dangerous mismatch in this codebase is colour order: **OpenCV decodes BGR, the
model was trained on RGB**. Getting that backwards produces confident, wrong
predictions on plausible-looking input. Tests guard it explicitly.

The contract mirrors the evaluation transform recorded in the research
checkpoint (`torchvision`: `Resize(256)` on the shorter side, `CenterCrop(224)`,
`ToTensor`, ImageNet `Normalize`), reimplemented here on OpenCV so the serving
path needs no torch or torchvision.

Reimplementation is not free: PIL's bilinear resize antialiases when
downscaling and OpenCV's `INTER_LINEAR` does not, so the two are far from
bit-identical. Measured over 40 test images, per-pixel divergence in normalised
units is substantial (mean worst-pixel 0.65-0.77 depending on interpolation).

What matters is whether that changes predictions, and measurement says it does
not: over 200 images the competition path agrees with the torchvision reference
on **200/200** predicted classes, with a mean worst-class probability difference
of 0.013. `INTER_AREA` is the default because it is the antialiasing-appropriate
choice for downscaling and measured marginally closer than the alternatives.
Full numbers in the Phase 2 parity results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

PREPROCESSING_VERSION = "phase2-imagenet-eval-1.0.0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessingError(ValueError):
    """Raised for input that cannot be preprocessed."""


@dataclass(frozen=True)
class PreprocessingContract:
    """Everything needed to reproduce the model's expected input tensor."""

    image_size: int = 224
    resize_shorter_side: int = 256
    colour_order: str = "RGB"
    scale: str = "0-1 after division by 255"
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "cv2.INTER_AREA"
    crop: str = "center"
    layout: str = "NCHW"
    version: str = PREPROCESSING_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        data["normalization_mean"] = list(self.normalization_mean)
        data["normalization_std"] = list(self.normalization_std)
        return data


DEFAULT_CONTRACT = PreprocessingContract()


def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise PreprocessingError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError(f"expected a 3-channel BGR image, got {image.shape!r}")
    if image.dtype != np.uint8:
        raise PreprocessingError(f"expected uint8 image, got dtype {image.dtype}")


def resize_shorter_side(image: np.ndarray, target: int, interpolation: int) -> np.ndarray:
    """Scale so the shorter side equals `target`, preserving aspect ratio."""
    height, width = image.shape[:2]
    if height <= width:
        new_height = target
        new_width = max(1, int(round(width * target / height)))
    else:
        new_width = target
        new_height = max(1, int(round(height * target / width)))
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Centre crop to `size` x `size`, matching torchvision's rounding."""
    height, width = image.shape[:2]
    if height < size or width < size:
        raise PreprocessingError(
            f"image {width}x{height} is smaller than the {size}x{size} crop"
        )
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image[top : top + size, left : left + size]


def preprocess_bgr(
    image: np.ndarray, contract: PreprocessingContract | None = None
) -> np.ndarray:
    """Turn a decoded BGR image into the model's NCHW input tensor.

    The input is a BGR array as OpenCV produces it. The **first** operation is
    the BGR to RGB conversion; everything downstream assumes RGB.

    Returns a float32 array of shape (1, 3, size, size). The source is not
    modified.
    """
    contract = contract or DEFAULT_CONTRACT
    _validate(image)

    interpolation = getattr(cv2, contract.interpolation.split(".")[-1])

    # BGR -> RGB first. This single line is the difference between a working
    # model and a confidently wrong one.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = resize_shorter_side(rgb, contract.resize_shorter_side, interpolation)
    cropped = center_crop(resized, contract.image_size)

    tensor = cropped.astype(np.float32) / 255.0
    mean = np.array(contract.normalization_mean, dtype=np.float32)
    std = np.array(contract.normalization_std, dtype=np.float32)
    tensor = (tensor - mean) / std

    # HWC -> CHW -> NCHW
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis, ...])


def contract_matches_checkpoint(
    contract: PreprocessingContract, checkpoint_preprocessing: dict
) -> tuple[bool, list[str]]:
    """Check the contract against the preprocessing recorded in a checkpoint.

    Returns (matches, differences). Used at load time so a checkpoint trained
    under different preprocessing cannot be served under this contract without
    the mismatch being visible.
    """
    differences: list[str] = []

    expected_size = int(checkpoint_preprocessing.get("image_size", contract.image_size))
    if expected_size != contract.image_size:
        differences.append(f"image_size {expected_size} != {contract.image_size}")

    expected_resize = int(
        checkpoint_preprocessing.get(
            "evaluation_resize_size", contract.resize_shorter_side
        )
    )
    if expected_resize != contract.resize_shorter_side:
        differences.append(
            f"resize {expected_resize} != {contract.resize_shorter_side}"
        )

    for name, expected, actual in (
        ("mean", checkpoint_preprocessing.get("normalization_mean"), contract.normalization_mean),
        ("std", checkpoint_preprocessing.get("normalization_std"), contract.normalization_std),
    ):
        if expected is None:
            continue
        if [round(float(v), 6) for v in expected] != [round(float(v), 6) for v in actual]:
            differences.append(f"{name} {list(expected)} != {list(actual)}")

    return (not differences), differences
