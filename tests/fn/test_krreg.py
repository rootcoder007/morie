"""Tests for morie.fn.krreg — Kernel ridge regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.krreg import krreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 50)
    result = krreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "alpha", "bandwidth", "penalty", "n_obs"):
        assert key in result


def test_interpolates_well():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x**2 + rng.normal(0, 0.05, 100)
    result = krreg(x, y, penalty=0.01)
    y_hat = np.asarray(result["y_hat"])
    assert np.corrcoef(y, y_hat)[0, 1] > 0.9


def test_nonpositive_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        krreg(np.ones(10), np.ones(10), penalty=0)


def test_alpha_length():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.1, 30)
    result = krreg(x, y)
    assert len(result["alpha"]) == 30


def test_prediction_is_the_kernel_expansion():
    import math

    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [1.0, 1.4, 0.9, 0.2, -0.3]
    h, lam = 0.7, 0.3
    r = krreg(x, y, [0.25, 1.75], bandwidth=h, penalty=lam)
    K = [[math.exp(-0.5 * ((a - b) / h) ** 2) / math.sqrt(2 * math.pi) for b in x] for a in x]
    al = r["alpha"]
    for i in range(5):
        assert sum(K[i][j] * al[j] for j in range(5)) + lam * al[i] == pytest.approx(y[i], rel=1e-10, abs=1e-12)
    want = [
        sum(math.exp(-0.5 * ((t - xj) / h) ** 2) / math.sqrt(2 * math.pi) * a for xj, a in zip(x, al))
        for t in (0.25, 1.75)
    ]
    assert r["y_hat"] == pytest.approx(want, rel=1e-12)
