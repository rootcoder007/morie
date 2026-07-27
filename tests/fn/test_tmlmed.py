"""Tests for tmlmed."""

import numpy as np
import pytest

from morie.fn.tmlmed import tmle_mediation


def test_tmlmed_basic():
    rng = np.random.default_rng(42)
    n = 3000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    M = 0.8 * A + 0.4 * W[:, 0] + rng.normal(scale=0.6, size=n)
    y = 0.5 * A + 1.0 * M + 0.3 * W[:, 0] + rng.normal(scale=0.6, size=n)
    out = tmle_mediation(y, A, M, W)
    assert out["total"] == pytest.approx(out["nde"] + out["nie"])
    assert out["total"] == pytest.approx(1.3, abs=0.3)


def test_tmlmed_edge():
    z = np.zeros(200)
    with pytest.raises(ValueError):
        tmle_mediation(np.arange(200.0), z, z)  # one arm
    with pytest.raises(ValueError):
        tmle_mediation(np.arange(200.0), (np.arange(200) % 2).astype(float), z[:10])
