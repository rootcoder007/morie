"""Tests for sntmod.sequential_target_models."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sntmod import sequential_target_models


def test_sntmod_basic():
    rng = np.random.default_rng(42)
    n = 3000
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = A1 + A2 + L2 + rng.normal(scale=0.5, size=n)
    hi = sequential_target_models(y, np.c_[A1, A2], np.c_[L1, L2], 1)
    lo = sequential_target_models(y, np.c_[A1, A2], np.c_[L1, L2], 0)
    assert hi["estimate"] - lo["estimate"] == pytest.approx(2.7, abs=0.25)
    assert hi["Qbar"].shape == (n,)


def test_sntmod_edge():
    with pytest.raises(ValueError):
        sequential_target_models([1.0, 2.0], [[0.5, 1]], [[0.0, 0.0]])  # non-binary
    with pytest.raises(ValueError):
        sequential_target_models([1.0], [[1, 0]], [[0.0]])  # shape mismatch
