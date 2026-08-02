"""Tests for tmltrt."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tmltrt import tmle_truncation


def _confounded(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(W @ np.array([1.0, -0.5, 0.3]))))
    A = (rng.random(n) < e).astype(float)
    y = 2.0 * A + W @ np.full(3, 1.0) + rng.normal(scale=0.5, size=n)
    return y, A, W


def test_tmltrt_basic():
    y, A, W = _confounded()
    out = tmle_truncation(y, A, W)
    assert out["ate"].size == out["eps"].size
    assert np.all(np.diff(out["n_truncated"]) >= 0)


def test_tmltrt_edge():
    y, A, W = _confounded()
    with pytest.raises(ValueError):
        tmle_truncation(y, A, W, eps_grid=[0.6])
    with pytest.raises(ValueError):
        tmle_truncation(y, A, W, eps_grid=[0.01])  # needs >= 2 values
