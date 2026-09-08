"""Tests for ascmcl.augmented_synthetic_control."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ascmcl import augmented_synthetic_control


def test_ascmcl_basic():
    # treated outside the donor hull: augmentation must beat plain SCM
    rng = np.random.default_rng(42)
    T, J, t0 = 40, 6, 25
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    Y0 = f @ rng.uniform(0, 1, size=(2, J)) + rng.normal(scale=0.1, size=(T, J))
    y1 = 1.5 * Y0[:, 0] - 0.2 * Y0[:, 1] + rng.normal(scale=0.1, size=T)
    y1[t0:] += 5.0
    result = augmented_synthetic_control(y1, Y0, t0)
    assert abs(result["att"] - 5.0) <= abs(result["att_scm"] - 5.0)
    assert result["att"] == pytest.approx(5.0, abs=1.0)
    assert result["att"] == pytest.approx(result["att_scm"] - result["correction"])


def test_ascmcl_edge():
    with pytest.raises(ValueError):
        augmented_synthetic_control(np.ones(10), np.ones((10, 3)), 5, ridge=0.0)
