"""Tests for morie.fn.pnreg — Penalized kernel regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.pnreg import pnreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 50)
    result = pnreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "bandwidth", "penalty", "n_obs"):
        assert key in result


def test_penalty_effect():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.5, 50)
    r1 = pnreg(x, y, penalty=0.01)
    r2 = pnreg(x, y, penalty=100.0)
    v1 = np.var(r1["y_hat"])
    v2 = np.var(r2["y_hat"])
    assert v2 <= v1 + 1e-6


def test_negative_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        pnreg(np.ones(10), np.ones(10), penalty=-1)


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        pnreg(np.ones(2), np.ones(2))


def test_penalised_local_linear_recomputed():
    """beta = (X'WX + lambda I)^-1 X'Wy with X = (1, x - x0)."""
    import math

    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    y = [1.0, 1.4, 0.9, 0.2, -0.3, 0.1]
    h, lam, x0 = 0.8, 0.5, 1.2
    w = [math.exp(-0.5 * ((a - x0) / h) ** 2) / math.sqrt(2 * math.pi) for a in x]
    d = [a - x0 for a in x]
    A = [
        [sum(w) + lam, sum(wi * di for wi, di in zip(w, d))],
        [sum(wi * di for wi, di in zip(w, d)), sum(wi * di * di for wi, di in zip(w, d)) + lam],
    ]
    bvec = [sum(wi * yi for wi, yi in zip(w, y)), sum(wi * di * yi for wi, di, yi in zip(w, d, y))]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    b0 = (A[1][1] * bvec[0] - A[0][1] * bvec[1]) / det
    r = pnreg(x, y, [x0], bandwidth=h, penalty=lam)
    assert r["y_hat"][0] == pytest.approx(b0, rel=1e-11)
