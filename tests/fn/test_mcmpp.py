"""Tests for mcmpp."""

from morie.fn import _array_core as np
import pytest

from morie.fn._stats_core import norm

from morie.fn.mcmpp import mcmcpack_irt


def test_mcmpp_basic():
    rng = np.random.default_rng(42)
    n, q = 30, 50
    x = np.linspace(-2, 2, n)
    beta = rng.normal(size=q)
    alpha = rng.normal(scale=0.5, size=q)
    V = (rng.random((n, q)) < norm.cdf(x[:, None] * beta - alpha)).astype(float)
    out = mcmcpack_irt(V, n_iter=300, burnin=100, seed=0, polarity_idx=0)
    assert np.corrcoef(out["ideal_points"], x)[0, 1] > 0.85
    assert out["ideal_points"][0] < 0


def test_mcmpp_edge():
    with pytest.raises(ValueError):
        mcmcpack_irt(np.full((4, 3), 2.0))  # non-binary
    with pytest.raises(ValueError):
        mcmcpack_irt(np.eye(4), n_iter=50, burnin=100)
