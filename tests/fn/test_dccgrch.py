"""Tests for dccgrch.dcc_garch (front-end to dccmd)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.dccgrch import dcc_garch
from morie.fn.dccmd import dcc_multivariate_garch


def _correlated(seed, n=600, rho=0.5):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    return rng.standard_normal((n, 2)) @ L.T


def test_dccgrch_delegates_to_the_real_dcc_engine():
    """Identical input must give identical estimates -- the front-end adds
    nothing. The placeholder it replaces returned spearmanr(X, X) = 1."""
    X = _correlated(0)
    a = dcc_garch(X)
    b = dcc_multivariate_garch(X)
    assert float(a["a"]) == pytest.approx(float(b["a"]), rel=1e-12)
    assert float(a["loglik"]) == pytest.approx(float(b["loglik"]), rel=1e-12)


def test_dccgrch_tracks_the_true_correlation():
    r = dcc_garch(_correlated(1, rho=0.5))
    R = np.asarray(r["conditional_correlation"], dtype=float)
    path = R[:, 0, 1] if R.ndim == 3 else np.atleast_1d(R)[..., 0]
    assert float(np.mean(path)) == pytest.approx(0.5, abs=0.1)


def test_dccgrch_rejects_what_the_engine_rejects():
    with pytest.raises(ValueError, match="n>=30"):
        dcc_garch(np.random.default_rng(0).standard_normal((10, 2)))
