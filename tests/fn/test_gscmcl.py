"""Tests for gscmcl.generalized_synthetic_control."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gscmcl import generalized_synthetic_control


def test_gscmcl_basic():
    rng = np.random.default_rng(42)
    T, J, t0, r = 50, 20, 30, 2
    f = np.cumsum(rng.normal(size=(T, r)), axis=0)
    Y0 = f @ rng.normal(size=(r, J)) + rng.normal(scale=0.2, size=(T, J))
    y1 = f @ np.array([2.0, -1.0]) + rng.normal(scale=0.2, size=T)
    y1[t0:] += 3.0
    result = generalized_synthetic_control(y1, Y0, t0, r=r)
    assert result["att"] == pytest.approx(3.0, abs=0.5)  # measured ~3.0
    assert result["gap"].shape == (T,)


def test_gscmcl_edge():
    with pytest.raises(ValueError):
        generalized_synthetic_control(np.ones(10), np.ones((10, 4)), 5, r=0)
    with pytest.raises(ValueError):
        generalized_synthetic_control(np.ones(10), np.ones((10, 4)), 1, r=1)  # t0 < 2
