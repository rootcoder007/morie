"""Tests for volrls."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volrls import vol_recursive_least_sq


def test_volrls_basic():
    r = np.array([1.0, 2.0] + [0.5] * 20)
    out = vol_recursive_least_sq(r, lam=0.9)
    assert out["sigma2"][1] == pytest.approx(0.9 * out["sigma2"][0] + 0.1 * 4.0)


def test_volrls_edge():
    with pytest.raises(ValueError):
        vol_recursive_least_sq([1.0])
    with pytest.raises(ValueError):
        vol_recursive_least_sq(np.ones(5), lam=1.0)
