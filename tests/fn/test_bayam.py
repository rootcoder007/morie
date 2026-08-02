"""Tests for bayam."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bayam import bayesian_am_scaling


def test_bayam_basic():
    rng = np.random.default_rng(42)
    n, q = 60, 5
    s = np.linspace(-1, 1, q)
    Z = (rng.normal(scale=0.4, size=n)[:, None]
         + rng.uniform(0.6, 1.4, size=n)[:, None] * s[None, :]
         + rng.normal(scale=0.1, size=(n, q)))
    out = bayesian_am_scaling(Z, n_iter=400, burnin=150, seed=0)
    s_norm = (s - s.mean()) / s.std()
    est = out["stimuli"]
    if np.corrcoef(est, s_norm)[0, 1] < 0:
        est = -est
    assert np.corrcoef(est, s_norm)[0, 1] > 0.98


def test_bayam_edge():
    with pytest.raises(ValueError):
        bayesian_am_scaling(np.ones((5, 2)))  # < 3 stimuli
    with pytest.raises(ValueError):
        bayesian_am_scaling(np.ones((5, 4)), n_iter=100, burnin=200)
