"""Tests for gctvc.g_computation_time_varying."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gctvc import g_computation_time_varying


def test_gctvc_basic():
    rng = np.random.default_rng(42)
    n = 3000
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = A1 + A2 + L2 + rng.normal(scale=0.5, size=n)
    result = g_computation_time_varying(y, np.c_[A1, A2], np.c_[L1, L2], n_mc=4000)
    assert result["estimate"] == pytest.approx(2.7, abs=0.25)  # measured ~2.68
    assert result["estimate"] == pytest.approx(result["EY_always"] - result["EY_never"])


def test_gctvc_edge():
    with pytest.raises(ValueError):
        g_computation_time_varying([1.0, 2.0], [[2, 0]], [[0.0, 0.0]])  # non-binary
