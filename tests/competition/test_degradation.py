"""Controlled degradation: reproducibility and source-image safety."""

from __future__ import annotations

import numpy as np
import pytest

from competition.vision.degradation import (
    BLUR_SIGMAS,
    DEGRADATION_VERSION,
    SUPPORTED_KINDS,
    DegradationError,
    DegradationSpec,
    add_glare,
    adjust_exposure,
    apply_degradation,
    gaussian_blur,
    occlude,
    reduce_contrast,
    standard_sweep,
)
from competition.vision.fixtures import checkerboard, textured_object


# --- required test 8: no source image is overwritten --------------------------


@pytest.mark.parametrize("kind,level", [
    ("none", 0.0),
    ("gaussian_blur", 3.0),
    ("underexpose", 0.4),
    ("overexpose", 2.0),
    ("reduce_contrast", 0.3),
    ("glare", 0.7),
    ("occlude", 0.25),
])
def test_apply_degradation_never_mutates_the_source(kind, level):
    source = textured_object()
    original = source.copy()
    degraded, _ = apply_degradation(source, DegradationSpec(kind=kind, level=level))
    assert np.array_equal(source, original), f"{kind} mutated its source image"
    assert degraded is not source


@pytest.mark.parametrize("transform,args", [
    (gaussian_blur, (2.0,)),
    (adjust_exposure, (0.5,)),
    (reduce_contrast, (0.5,)),
    (add_glare, (0.6,)),
    (occlude, (0.2,)),
])
def test_individual_transforms_never_mutate_the_source(transform, args):
    source = checkerboard()
    original = source.copy()
    transform(source, *args)
    assert np.array_equal(source, original)


def test_identity_levels_return_an_independent_copy():
    """Even a no-op must return a new array, or callers can alias the source."""
    source = textured_object()
    degraded, _ = apply_degradation(source, DegradationSpec(kind="gaussian_blur", level=0.0))
    assert np.array_equal(degraded, source)
    assert degraded is not source
    degraded[0, 0] = [1, 2, 3]
    assert not np.array_equal(degraded, source)


# --- reproducibility ----------------------------------------------------------


@pytest.mark.parametrize("kind,level", [
    ("gaussian_blur", 2.0),
    ("underexpose", 0.4),
    ("overexpose", 2.0),
    ("reduce_contrast", 0.3),
    ("glare", 0.7),
    ("occlude", 0.25),
])
def test_same_spec_and_seed_reproduces_identical_output(kind, level):
    source = textured_object()
    spec = DegradationSpec(kind=kind, level=level, seed=42)
    first, _ = apply_degradation(source, spec)
    second, _ = apply_degradation(source, spec)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind", ["glare", "occlude"])
def test_seed_changes_spatial_placement(kind):
    """Seeded transforms must actually respond to the seed."""
    source = textured_object()
    a, _ = apply_degradation(source, DegradationSpec(kind=kind, level=0.5, seed=1))
    b, _ = apply_degradation(source, DegradationSpec(kind=kind, level=0.5, seed=999))
    assert not np.array_equal(a, b)


def test_metadata_records_the_specification():
    _, metadata = apply_degradation(
        textured_object(), DegradationSpec(kind="gaussian_blur", level=2.5, seed=7)
    )
    assert metadata["kind"] == "gaussian_blur"
    assert metadata["level"] == 2.5
    assert metadata["seed"] == 7
    assert metadata["degradation_version"] == DEGRADATION_VERSION


# --- transform semantics ------------------------------------------------------


def test_blur_ladder_starts_undegraded():
    assert BLUR_SIGMAS[0] == 0.0


def test_underexposure_darkens_and_overexposure_brightens():
    image = textured_object()
    base = float(image.mean())
    assert float(adjust_exposure(image, 0.4).mean()) < base
    assert float(adjust_exposure(image, 2.0).mean()) > base


def test_contrast_reduction_narrows_the_value_range():
    image = checkerboard()
    original_spread = float(image.max()) - float(image.min())
    reduced = reduce_contrast(image, 0.2)
    assert float(reduced.max()) - float(reduced.min()) < original_spread


def test_glare_only_brightens():
    image = textured_object()
    glared = add_glare(image, 0.8, seed=3)
    assert np.all(glared >= image)
    assert float(glared.mean()) > float(image.mean())


def test_occlusion_covers_approximately_the_requested_area():
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    occluded = occlude(image, 0.25, seed=5)
    black_fraction = float(np.count_nonzero(np.all(occluded == 0, axis=2))) / (200 * 200)
    assert 0.20 <= black_fraction <= 0.30


# --- specification validation -------------------------------------------------


def test_unknown_kind_is_rejected():
    with pytest.raises(DegradationError):
        apply_degradation(textured_object(), DegradationSpec(kind="sharpen", level=1.0))


@pytest.mark.parametrize("transform,bad", [
    (gaussian_blur, -1.0),
    (adjust_exposure, 0.0),
    (reduce_contrast, 1.5),
    (add_glare, 1.5),
    (occlude, 1.0),
])
def test_out_of_range_parameters_are_rejected(transform, bad):
    with pytest.raises(DegradationError):
        transform(textured_object(), bad)


def test_empty_image_is_rejected():
    with pytest.raises(DegradationError):
        apply_degradation(
            np.zeros((0, 0, 3), dtype=np.uint8),
            DegradationSpec(kind="gaussian_blur", level=1.0),
        )


# --- sweep --------------------------------------------------------------------


def test_standard_sweep_covers_exactly_the_phase1_ladder():
    """Frozen: the Phase 1 findings were measured over these families only."""
    from competition.vision.degradation import PHASE1_SWEEP_KINDS

    kinds = {spec.kind for spec in standard_sweep()}
    assert kinds == set(PHASE1_SWEEP_KINDS)


def test_every_supported_kind_is_exercised_by_some_ladder():
    """No transform may exist without a sweep that measures it."""
    from competition.vision.degradation import subject_scale_sweep

    exercised = {spec.kind for spec in standard_sweep()}
    exercised |= {spec.kind for spec in subject_scale_sweep()}
    assert exercised == set(SUPPORTED_KINDS)


def test_standard_sweep_is_reproducible():
    assert [s.to_dict() for s in standard_sweep(seed=11)] == [
        s.to_dict() for s in standard_sweep(seed=11)
    ]


def test_standard_sweep_specs_all_apply_cleanly():
    image = textured_object()
    for spec in standard_sweep():
        degraded, metadata = apply_degradation(image, spec)
        assert degraded.shape == image.shape
        assert degraded.dtype == np.uint8
        assert metadata["kind"] == spec.kind
