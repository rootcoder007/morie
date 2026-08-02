"""Tests for gb1251.gibbons_partial_tau."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1251 import gibbons_partial_tau


def test_gb1251_basic():
    rng = np.random.default_rng(42)
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.4, size=n)
    y = z + rng.normal(scale=0.4, size=n)
    out = gibbons_partial_tau(x, y, z)
    assert out["tau_xy"] > 0.4
    assert abs(out["partial_tau"]) < 0.5 * out["tau_xy"]  # shrinks sharply


def test_gb1251_edge():
    with pytest.raises(ValueError):
        gibbons_partial_tau([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])  # n < 4
    with pytest.raises(ValueError):
        gibbons_partial_tau([1.0] * 5, [1.0] * 5, [1.0] * 4)  # length mismatch
