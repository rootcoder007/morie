"""Tests for medML.ml_mediation_dml."""

import numpy as np
import pytest

from morie.fn.medML import ml_mediation_dml


def _confounded(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, 5))
    x = C @ np.array([0.5, -0.3, 0.2, 0.0, 0.1]) + rng.normal(size=n)
    m = 0.8 * x + C @ np.full(5, 0.3) + rng.normal(scale=0.6, size=n)
    y = 0.7 * x + 1.5 * m + C @ np.full(5, -0.2) + rng.normal(scale=0.6, size=n)
    return x, m, y, C


def test_medML_basic():
    out = ml_mediation_dml(*_confounded(), n_folds=5, seed=0)
    assert out["indirect"] == pytest.approx(1.2, abs=0.15)
    assert out["direct"] == pytest.approx(0.7, abs=0.15)


def test_medML_edge():
    x, m, y, C = _confounded()
    with pytest.raises(ValueError):
        ml_mediation_dml(x, m, y, C, n_folds=1)
    with pytest.raises(ValueError):
        ml_mediation_dml(x[:10], m, y, C)  # length mismatch
