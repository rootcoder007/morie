"""Tests for dccmd.dcc_multivariate_garch."""

import numpy as np
import pytest

from morie.fn.dccmd import dcc_multivariate_garch


def _correlated(seed, n=1200, rho=0.6):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    return rng.standard_normal((n, 2)) @ L.T


def test_dccmd_recovers_a_constant_correlation():
    """Constant-correlation data: the conditional correlation path must
    hover around rho = 0.6 and the DCC dynamics must not explode."""
    r = dcc_multivariate_garch(_correlated(0))
    R = np.asarray(r["conditional_correlation"], dtype=float)
    path = R[:, 0, 1] if R.ndim == 3 else np.atleast_1d(R)[..., 0]
    assert float(np.mean(path)) == pytest.approx(0.6, abs=0.08)
    a, b = float(r["a"]), float(r["b"])
    assert a >= 0 and b >= 0 and a + b < 1


def test_dccmd_independent_series_give_near_zero_correlation():
    r = dcc_multivariate_garch(_correlated(1, rho=0.0))
    R = np.asarray(r["conditional_correlation"], dtype=float)
    path = R[:, 0, 1] if R.ndim == 3 else np.atleast_1d(R)[..., 0]
    assert abs(float(np.mean(path))) < 0.1


def test_dccmd_unconditional_matrix_is_a_correlation_matrix():
    r = dcc_multivariate_garch(_correlated(2))
    Q = np.asarray(r["unconditional_correlation"], dtype=float)
    np.testing.assert_allclose(np.diag(Q), 1.0, atol=0.05)
    assert abs(Q[0, 1] - 0.6) < 0.1


def test_dccmd_rejects_short_or_univariate_input():
    with pytest.raises(ValueError, match="n>=30"):
        dcc_multivariate_garch(np.random.default_rng(0).standard_normal((10, 2)))
    with pytest.raises(ValueError, match="k>=2"):
        dcc_multivariate_garch(np.random.default_rng(0).standard_normal((100, 1)))
