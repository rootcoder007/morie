"""Tests for survcfg."""

import numpy as np
import pytest

from morie.fn.csfgrf import causal_survival_forest
from morie.fn.survcfg import causal_survival_forest_grf


def _surv(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    t_event = rng.exponential(np.exp(0.5 * D))
    cens = rng.exponential(4.0, size=n)
    return np.minimum(t_event, cens), (t_event <= cens).astype(float), D, X


def test_survcfg_basic():
    time, event, D, X = _surv()
    a = causal_survival_forest_grf(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    b = causal_survival_forest(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    assert a["ate"] == pytest.approx(b["ate"])


def test_survcfg_edge():
    time, event, D, X = _surv()
    with pytest.raises(ValueError):
        causal_survival_forest_grf(time, np.full(time.size, 0.5), D, X)  # non-binary event
