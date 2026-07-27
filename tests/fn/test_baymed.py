"""Tests for baymed.bayes_mediation."""

import numpy as np
import pytest

from morie.fn.baymed import bayes_mediation


def _simple(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_baymed_basic():
    out = bayes_mediation(*_simple(), n_draws=2000, seed=0)
    assert out["indirect_mean"] == pytest.approx(1.2, abs=0.15)
    lo, hi = out["indirect_ci"]
    assert lo < 1.2 < hi
    assert out["draws"].size == 2000


def test_baymed_edge():
    with pytest.raises(ValueError):
        bayes_mediation(*_simple(), prior_sd=0.0)
    with pytest.raises(ValueError):
        bayes_mediation(*_simple(), n_draws=10)
