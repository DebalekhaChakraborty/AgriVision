"""Preprocessing contract: colour order, geometry, determinism."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from competition.models.preprocessing import (
    DEFAULT_CONTRACT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessingContract,
    PreprocessingError,
    center_crop,
    contract_matches_checkpoint,
    preprocess_bgr,
    resize_shorter_side,
)
from competition.vision.fixtures import checkerboard, textured_object


# --- required test 6: BGR/RGB conversion is correct ---------------------------


def test_bgr_to_rgb_conversion_is_applied():
    """A pure-blue BGR image must land in the RED channel of the tensor.

    This is the single most dangerous silent bug available here: OpenCV decodes
    BGR, the model expects RGB, and getting it backwards produces confident
    wrong predictions on input that looks perfectly normal.
    """
    blue_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255  # BGR channel 0 is blue

    tensor = preprocess_bgr(blue_bgr)

    # Undo normalisation to recover 0-1 RGB values.
    channel_values = [
        float((tensor[0, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]).mean())
        for c in range(3)
    ]
    # RGB order: index 2 (blue) should be saturated, index 0 (red) empty.
    assert channel_values[2] > 0.95, f"blue did not land in the B slot: {channel_values}"
    assert channel_values[0] < 0.05, f"blue leaked into the R slot: {channel_values}"


def test_red_bgr_lands_in_red_channel():
    red_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255  # BGR channel 2 is red
    tensor = preprocess_bgr(red_bgr)
    values = [
        float((tensor[0, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]).mean())
        for c in range(3)
    ]
    assert values[0] > 0.95
    assert values[2] < 0.05


def test_swapping_channels_changes_the_tensor():
    """Sanity: if BGR/RGB did not matter, this test could not fail."""
    image = textured_object()
    swapped = image[:, :, ::-1].copy()
    assert not np.allclose(preprocess_bgr(image), preprocess_bgr(swapped))


# --- required test 7: preprocessing contract is stable ------------------------


def test_contract_values_are_pinned():
    contract = DEFAULT_CONTRACT
    assert contract.image_size == 224
    assert contract.resize_shorter_side == 256
    assert contract.colour_order == "RGB"
    assert contract.layout == "NCHW"
    assert contract.normalization_mean == IMAGENET_MEAN
    assert contract.normalization_std == IMAGENET_STD
    assert contract.version == "phase2-imagenet-eval-1.0.0"


def test_output_shape_and_dtype():
    tensor = preprocess_bgr(textured_object())
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_preprocessing_is_deterministic():
    image = textured_object()
    assert np.array_equal(preprocess_bgr(image), preprocess_bgr(image))


def test_preprocessing_does_not_mutate_the_source():
    image = textured_object()
    before = image.copy()
    preprocess_bgr(image)
    assert np.array_equal(image, before)


def test_resize_preserves_aspect_ratio_on_the_shorter_side():
    tall = np.zeros((400, 200, 3), dtype=np.uint8)
    resized = resize_shorter_side(tall, 256, cv2.INTER_AREA)
    assert min(resized.shape[:2]) == 256
    assert resized.shape[0] == 512  # 400/200 preserved

    wide = np.zeros((200, 400, 3), dtype=np.uint8)
    resized = resize_shorter_side(wide, 256, cv2.INTER_AREA)
    assert min(resized.shape[:2]) == 256
    assert resized.shape[1] == 512


def test_center_crop_is_centred():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[138:162, 138:162] = 255  # centred 24x24 block
    cropped = center_crop(image, 224)
    assert cropped.shape[:2] == (224, 224)
    centre = cropped[100:124, 100:124]
    assert centre.mean() > 200


def test_crop_larger_than_image_is_rejected():
    with pytest.raises(PreprocessingError):
        center_crop(np.zeros((64, 64, 3), dtype=np.uint8), 224)


def test_small_image_is_upscaled_then_croppable():
    """Resize happens before crop, so a small image is still usable."""
    tensor = preprocess_bgr(np.full((64, 64, 3), 120, dtype=np.uint8))
    assert tensor.shape == (1, 3, 224, 224)


@pytest.mark.parametrize("bad", [
    None,
    "not an image",
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((32, 32), dtype=np.uint8),
    np.zeros((32, 32, 3), dtype=np.float32),
])
def test_invalid_input_is_rejected(bad):
    with pytest.raises(PreprocessingError):
        preprocess_bgr(bad)


def test_normalisation_is_applied():
    """A mid-grey image should land near zero after ImageNet normalisation."""
    grey = np.full((256, 256, 3), 124, dtype=np.uint8)  # ~0.486, close to the mean
    tensor = preprocess_bgr(grey)
    assert abs(float(tensor.mean())) < 0.3


# --- checkpoint agreement -----------------------------------------------------


def test_contract_matches_a_conforming_checkpoint():
    matches, differences = contract_matches_checkpoint(
        DEFAULT_CONTRACT,
        {
            "image_size": 224,
            "evaluation_resize_size": 256,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
    )
    assert matches
    assert not differences


def test_contract_mismatch_is_reported_not_ignored():
    matches, differences = contract_matches_checkpoint(
        DEFAULT_CONTRACT,
        {
            "image_size": 299,
            "evaluation_resize_size": 320,
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
        },
    )
    assert not matches
    assert len(differences) == 4


def test_custom_contract_changes_output_size():
    tensor = preprocess_bgr(
        checkerboard(), PreprocessingContract(image_size=128, resize_shorter_side=160)
    )
    assert tensor.shape == (1, 3, 128, 128)
