"""Tests for isotr."""

from morie.fn import _array_core as np
import pytest

from morie.fn.isotr import isotonic_regression_disparity


def test_isotr_basic():
    out = isotonic_regression_disparity([1.0, 3.0, 2.0, 4.0], [0, 1, 2, 3])
    assert out["disparities"] == pytest.approx([1.0, 2.5, 2.5, 4.0])
    assert np.all(np.diff(out["sorted_fit"]) >= -1e-12)


def test_isotr_edge():
    with pytest.raises(ValueError):
        isotonic_regression_disparity([1.0], [0])  # single pair
    with pytest.raises(ValueError):
        isotonic_regression_disparity([1.0, 2.0], [0])  # length mismatch
