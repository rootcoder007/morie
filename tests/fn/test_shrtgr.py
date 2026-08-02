"""Tests for shrtgr.shrinkage_propensity."""

from morie.fn import _array_core as np
import pytest

from morie.fn.shrtgr import shrinkage_propensity


def test_shrtgr_basic():
    rng = np.random.default_rng(42)
    n = 200
    h = rng.normal(size=n)
    A = (h > 0).astype(float)  # perfectly separated
    tight = shrinkage_propensity(A, h, prior_sd=0.5)
    loose = shrinkage_propensity(A, h, prior_sd=100.0)
    assert tight["ps_min"] > loose["ps_min"]
    assert abs(tight["coefficients"][1]) < abs(loose["coefficients"][1])


def test_shrtgr_edge():
    with pytest.raises(ValueError):
        shrinkage_propensity([1, 0], [1.0, 2.0], prior_sd=0.0)
    with pytest.raises(ValueError):
        shrinkage_propensity([0.5, 1.0], [1.0, 2.0])  # non-binary A
