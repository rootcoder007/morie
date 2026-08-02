"""Tests for csurv2."""

from morie.fn import _array_core as np
import pytest

from morie.fn.csurv2 import causal_survival_blp


def _surv(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    t_event = rng.exponential(np.exp(0.5 * D))
    cens = rng.exponential(4.0, size=n)
    return np.minimum(t_event, cens), (t_event <= cens).astype(float), D, X


def test_csurv2_basic():
    time, event, D, X = _surv()
    out = causal_survival_blp(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    assert np.isfinite(out["beta"])
    assert out["horizon"] > 0


def test_csurv2_edge():
    time, event, D, X = _surv()
    with pytest.raises(ValueError):
        causal_survival_blp(time, event, np.full(time.size, 0.5), X)  # non-binary D
