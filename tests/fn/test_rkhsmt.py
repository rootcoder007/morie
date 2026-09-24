"""Tests for rkhsmt.rkhs_multitrait."""

import math

from morie.fn import _array_core as np

from morie.fn.rkhsmt import rkhs_multitrait


def _make_psd(rng, J, r):
    """Build a J-by-J symmetric positive-(semi)definite matrix via A @ A.T + ridge."""
    A = rng.normal(0, 1, (J, r))
    K = [[sum(A[i][k] * A[j][k] for k in range(r)) for j in range(J)]
         for i in range(J)]
    # small diagonal ridge to make it strictly positive definite
    for i in range(J):
        K[i][i] += 1e-6
    return K


def test_rkhsmt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    J, nT = 40, 3
    Y = rng.normal(0, 1, (J, nT))
    # Build a symmetric positive-definite kernel matrix of order J
    K = _make_psd(rng, J, J)
    result = rkhs_multitrait(Y, K, n_iter=50)
    # RichResult-style object exposing the keys from the docstring
    assert "estimate" in result
    assert "gebv" in result
    assert "b1" in result
    assert "Sigma_T" in result
    assert "R" in result
    assert "mu" in result
    assert "beta" in result
    # Shape checks for the J-by-n_T outputs
    gebv = result["gebv"]
    assert len(gebv) == J
    assert len(gebv[0]) == nT
    b1 = result["b1"]
    assert len(b1) == J
    assert len(b1[0]) == nT
    # Shape checks for the n_T-by-n_T covariance matrices
    ST = result["Sigma_T"]
    assert len(ST) == nT
    assert len(ST[0]) == nT
    R = result["R"]
    assert len(R) == nT
    assert len(R[0]) == nT
    # mu is the n_T trait intercepts
    mu = result["mu"]
    assert len(mu) == nT
    # estimate is the scalar mean of fitted genomic values
    assert math.isfinite(result["estimate"])
    # beta is empty when X is None
    assert len(result["beta"]) == 0


def test_rkhsmt_edge():
    """Test edge case with a small J and a positive-definite kernel."""
    rng = np.random.default_rng(43)
    J, nT = 10, 2
    Y = rng.normal(0, 1, (J, nT))
    K = _make_psd(rng, J, J)
    result = rkhs_multitrait(Y, K, n_iter=5)
    assert "estimate" in result
    assert "gebv" in result
    assert "b1" in result
    assert "Sigma_T" in result
    assert "R" in result
    assert "mu" in result
    assert "beta" in result
    gebv = result["gebv"]
    assert len(gebv) == J
    assert len(gebv[0]) == nT
    assert math.isfinite(result["estimate"])
    # beta is empty when X is None
    assert len(result["beta"]) == 0
