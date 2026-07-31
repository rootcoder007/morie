"""Book-certified tests for the Schabenberger & Gotway Ch. 6 family.

Schabenberger, O. & Gotway, C. A. (2005). Ch. 6.
"""

import numpy as np
import pytest

from morie.fn.spgls import schabenberger_gls_spatial as gls
from morie.fn.spicar import schabenberger_icar_prior as icar_prior
from morie.fn.spsar import schabenberger_sar_model as sar
from morie.fn.sarla import spatial_ar_lag
from morie.fn.spcar import schabenberger_car_model as car
from morie.fn.sgcar import conditional_autoregressive
from morie.fn.spgwr import schabenberger_gwr as gwr
from morie.fn.gwreg import geographically_weighted_regression


def _reg(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.random(n)])
    return X, X @ np.array([2.0, -1.0]) + rng.normal(0, 0.3, n)


def _adjacency(n=12, seed=1):
    rng = np.random.default_rng(seed)
    W = (rng.random((n, n)) < 0.3).astype(float)
    W = ((W + W.T) > 0).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def test_gls_reduces_to_ols_when_sigma_is_scalar():
    """Sigma = sigma^2 I makes GLS exactly OLS (Sec 6.2.3).

    An implementation that whitens the wrong way round fails here.
    """
    X, y = _reg()
    r = gls(X, y, 4.0 * np.eye(len(y)))
    np.testing.assert_allclose(r["beta"], r["beta_ols"], atol=1e-12)


def test_gls_differs_from_ols_under_correlation():
    X, y = _reg()
    n = len(y)
    S = np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 5.0)
    r = gls(X, y, S)
    assert not np.allclose(r["beta"], r["beta_ols"])


def test_naive_ols_standard_errors_understate_under_correlation():
    """OLS stays unbiased but its usual SEs are computed as if
    Sigma = sigma^2 I, so they are wrong (Sec 6.2.3)."""
    X, y = _reg()
    n = len(y)
    S = np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 5.0)
    r = gls(X, y, S)
    assert np.all(r["se_ols_naive"] < r["se_ols_correct"])


def test_gls_closed_form():
    """beta = (X' S^-1 X)^-1 X' S^-1 Z, Var = (X' S^-1 X)^-1."""
    X, y = _reg(n=20, seed=4)
    n = len(y)
    S = np.eye(n) + 0.3 * np.ones((n, n))
    Si = np.linalg.inv(S)
    expected = np.linalg.inv(X.T @ Si @ X) @ (X.T @ Si @ y)
    r = gls(X, y, S)
    np.testing.assert_allclose(r["beta"], expected, rtol=1e-10)
    np.testing.assert_allclose(r["vcov"], np.linalg.inv(X.T @ Si @ X), rtol=1e-10)


def test_gls_input_validation():
    X, y = _reg(n=10)
    with pytest.raises(ValueError, match="same number of rows"):
        gls(X, y[:-1])
    with pytest.raises(ValueError, match="to match the data"):
        gls(X, y, np.eye(3))


def test_icar_precision_is_the_graph_laplacian_and_is_improper():
    """Q = D - W, and Q1 = 0 exactly: the prior is improper (Sec 6.4.3)."""
    W = _adjacency()
    n = W.shape[0]
    r = icar_prior(W)
    np.testing.assert_allclose(r["Q"], np.diag(W.sum(axis=1)) - W, atol=1e-12)
    np.testing.assert_allclose(r["Q"] @ np.ones(n), np.zeros(n), atol=1e-12)
    assert r["is_improper"]
    assert r["rank"] == n - r["n_components"]


def test_icar_rank_deficiency_counts_connected_components():
    """Two disjoint blocks leave a rank deficiency of two."""
    blk = np.array([[0.0, 1.0], [1.0, 0.0]])
    W = np.zeros((4, 4))
    W[:2, :2] = blk
    W[2:, 2:] = blk
    r = icar_prior(W)
    assert r["n_components"] == 2


def test_icar_conditional_variance_is_tau2_over_degree():
    W = _adjacency()
    tau2 = 2.5
    r = icar_prior(W, tau2=tau2)
    d = W.sum(axis=1)
    np.testing.assert_allclose(r["conditional_variances"][d > 0],
                               tau2 / d[d > 0], rtol=1e-12)


def test_icar_input_validation():
    with pytest.raises(ValueError, match="square"):
        icar_prior(np.ones((2, 3)))
    with pytest.raises(ValueError, match="`tau2` must be"):
        icar_prior(np.eye(3), tau2=0.0)


def _spatial_case(n=25, seed=3):
    rng = np.random.default_rng(seed)
    W = (rng.random((n, n)) < 0.25).astype(float)
    W = ((W + W.T) > 0).astype(float)
    np.fill_diagonal(W, 0.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    X = np.column_stack([np.ones(n), rng.random(n)])
    y = 0.4 * (W @ rng.random(n)) + X @ np.array([1.0, -0.5]) + rng.normal(0, 0.3, n)
    return X, y, W, rng.random((n, 2))


def _same(a, b):
    """Array-aware deep compare.

    Not str(): RichResult is a dict subclass whose __str__ is empty
    unless it was given a title, so string equality passes vacuously.
    """
    if hasattr(a, "statistic") and not isinstance(a, dict):
        return a.statistic == b.statistic
    a, b = dict(a), dict(b)
    if set(a) != set(b):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            if not np.array_equal(np.asarray(av, dtype=object),
                                  np.asarray(bv, dtype=object)):
                return False
        elif av != bv:
            return False
    return True


def test_same_helper_actually_discriminates():
    """Guard: a comparator that returns True for everything is useless."""
    X, y, W, coords = _spatial_case()
    a = gwr(X, y, coords)
    b = gwr(X, y, coords + 1.0)
    assert not _same(a, b)


def test_sar_delegates_to_spatial_ar_lag():
    X, y, W, _ = _spatial_case()
    got = sar(X, y, W)
    assert _same(got, spatial_ar_lag(X, y, W))
    assert len(dict(got)) > 0


def test_car_delegates_to_conditional_autoregressive():
    _, y, W, _ = _spatial_case()
    assert car(y, W).statistic == conditional_autoregressive(y, W).statistic


def test_gwr_delegates_to_geographically_weighted_regression():
    X, y, _, coords = _spatial_case()
    got = gwr(X, y, coords)
    assert _same(got, geographically_weighted_regression(X, y, coords))
    assert len(dict(got)) > 0
