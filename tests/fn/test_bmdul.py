"""Tests for bmdul."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bmdul import bayesian_mds_unfolding


def test_bmdul_basic():
    rng = np.random.default_rng(42)
    n, q = 20, 5
    Xt = rng.uniform(-1, 1, size=(n, 1))
    Yt = np.linspace(-1, 1, q)[:, None]
    T = 4.0 - (Xt[:, None, 0] - Yt[None, :, 0]) ** 2 + rng.normal(scale=0.1, size=(n, q))
    out = bayesian_mds_unfolding(T, n_dims=1, n_iter=350, burnin=150, seed=0)
    assert abs(np.corrcoef(out["stimuli"][:, 0], Yt[:, 0])[0, 1]) > 0.9


def test_bmdul_edge():
    T = np.ones((6, 4))
    with pytest.raises(ValueError):
        bayesian_mds_unfolding(T, n_dims=0)
    with pytest.raises(ValueError):
        bayesian_mds_unfolding(np.full((6, 4), np.nan))  # all missing
