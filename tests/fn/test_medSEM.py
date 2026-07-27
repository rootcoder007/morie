"""Tests for medSEM.sem_mediation."""

import numpy as np
import pytest

from morie.fn.medSEM import sem_mediation


def test_medSEM_basic():
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.normal(size=n)
    M = 0.8 * X + rng.normal(scale=0.6, size=n)
    Y = 0.5 * X + 1.2 * M + rng.normal(scale=0.6, size=n)
    out = sem_mediation({"M": ["X"], "Y": ["X", "M"]}, {"X": X, "M": M, "Y": Y})
    assert out["paths"]["X->M->Y"] == pytest.approx(0.96, abs=0.08)
    assert out["total_effects"]["X"] == pytest.approx(sum(out["paths"].values()))


def test_medSEM_edge():
    with pytest.raises(ValueError):
        sem_mediation({}, {"X": [1.0, 2.0]})
    with pytest.raises(ValueError):
        sem_mediation({"Y": ["Z"]}, {"Y": [1.0, 2.0, 3.0]})  # missing Z
