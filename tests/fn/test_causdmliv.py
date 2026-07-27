"""Tests for causdmliv.causal_dml_iv."""

import numpy as np
import pytest

from morie.fn.causdmliv import causal_dml_iv


def _iv(seed=42, n=3000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    d = 0.8 * z + u + X @ np.full(4, 0.3) + rng.normal(scale=0.5, size=n)
    y = 1.5 * d + 2.0 * u + X @ np.full(4, -0.2) + rng.normal(scale=0.5, size=n)
    return y, d, z, X


def test_causdmliv_basic():
    y, d, z, X = _iv()
    out = causal_dml_iv(y, d, z, X, n_folds=5, seed=0)
    assert out["theta"] == pytest.approx(1.5, abs=0.2)
    assert out["se"] > 0


def test_causdmliv_edge():
    y, d, z, X = _iv()
    with pytest.raises(ValueError):
        causal_dml_iv(y, d, z, X, n_folds=1)
    with pytest.raises(ValueError):
        causal_dml_iv(y[:10], d, z, X)  # length mismatch
