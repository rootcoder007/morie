"""Tests for vecmf.vecm."""

import numpy as np
import pytest

from morie.fn.vecmf import vecm


def _cointegrated(seed, n=500):
    rng = np.random.default_rng(seed)
    x1 = np.cumsum(rng.standard_normal(n))
    x2 = x1 + rng.standard_normal(n) * 0.5
    return np.column_stack([x1, x2])


def test_vecmf_beta_spans_the_true_cointegrating_vector():
    """The DGP's equilibrium is x1 - x2 = stationary, so beta normalised
    on its first entry must be close to (1, -1) (Johansen 1995)."""
    r = vecm(_cointegrated(1), k_ar=1, coint_rank=1)
    beta = np.asarray(r["beta"], dtype=float).ravel()
    beta = beta / beta[0]
    assert beta[1] == pytest.approx(-1.0, abs=0.05)


def test_vecmf_alpha_pulls_toward_equilibrium():
    """The loading on the error-correction term must stabilise the system:
    the disequilibrium z = x1 - x2 mean-reverts, so alpha[1] - alpha[0]
    (the effect on d(x2 - x1)) has the sign that shrinks z."""
    r = vecm(_cointegrated(2), k_ar=1, coint_rank=1)
    alpha = np.asarray(r["alpha"], dtype=float).ravel()
    beta = np.asarray(r["beta"], dtype=float).ravel()
    # Error-correction stability: alpha' beta < 0 in the scalar rank-1 case.
    assert float(alpha @ beta) < 0


def test_vecmf_sigma_is_positive_definite():
    r = vecm(_cointegrated(3), k_ar=1, coint_rank=1)
    S = np.asarray(r["Sigma"], dtype=float)
    assert np.all(np.linalg.eigvalsh(S) > 0)


def test_vecmf_rejects_bad_rank_and_short_series():
    data = _cointegrated(4)
    with pytest.raises(ValueError, match="rank"):
        vecm(data, coint_rank=3)
    with pytest.raises(ValueError, match="T>=20"):
        vecm(data[:10], coint_rank=1)
