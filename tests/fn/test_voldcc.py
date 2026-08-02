"""Tests for voldcc.vol_dcc_garch (Engle 2002).

Properties come from equations (9) and (10) of the rmgarch model
reference: the Q_t recursion with a + b < 1, and R_t obtained by
rescaling Q_t to unit diagonal.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.dccmd import dcc_multivariate_garch
from morie.fn.voldcc import vol_dcc_garch


def _panel(seed=3, n=400, rho=0.7):
    """Two GARCH-like series with a known constant correlation."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    h = np.empty((n, 2))
    r = np.empty((n, 2))
    for j in range(2):
        h[0, j] = 1.0
        r[0, j] = np.sqrt(h[0, j]) * z[0, j]
        for t in range(1, n):
            h[t, j] = 0.05 + 0.10 * r[t - 1, j] ** 2 + 0.85 * h[t - 1, j]
            r[t, j] = np.sqrt(h[t, j]) * z[t, j]
    return r


def test_parameters_satisfy_the_stationarity_constraint():
    res = vol_dcc_garch(_panel())
    a, b = res["a"], res["b"]
    assert a >= 0 and b >= 0
    assert a + b < 1, "a + b < 1 is required for stationarity and positive definiteness"


def test_conditional_correlations_have_unit_diagonal():
    """Equation (10) rescales Q_t by its own diagonal, so diag(R_t) == 1."""
    res = vol_dcc_garch(_panel())
    R = res["conditional_correlation"]
    assert R.shape == (400, 2, 2)
    assert np.allclose(R[:, 0, 0], 1.0)
    assert np.allclose(R[:, 1, 1], 1.0)
    assert np.allclose(R[:, 0, 1], R[:, 1, 0])
    assert np.all(np.abs(R[:, 0, 1]) <= 1.0 + 1e-9)


def test_unconditional_correlation_is_symmetric():
    Q = vol_dcc_garch(_panel())["Q_bar"]
    assert np.allclose(Q, Q.T)


def test_tracks_the_generating_correlation_on_average():
    res = vol_dcc_garch(_panel(rho=0.7))
    assert float(np.mean(res["conditional_correlation"][:, 0, 1])) == pytest.approx(0.7, abs=0.15)


def test_is_a_front_end_over_dccmd():
    """This module must not carry a second copy of the estimator."""
    X = _panel()
    assert vol_dcc_garch(X)["a"] == dcc_multivariate_garch(X)["a"]


def test_short_names_alias_the_long_ones():
    res = vol_dcc_garch(_panel())
    assert res["ll"] == res["loglik"]
    assert np.array_equal(res["Q_bar"], res["unconditional_correlation"])
    assert np.allclose(res["sigmas"], np.sqrt(res["conditional_variance"]))


def test_rejects_single_series():
    with pytest.raises(ValueError, match="k>=2"):
        vol_dcc_garch(np.random.default_rng(0).normal(0, 1, 100))
