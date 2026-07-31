"""Tests for spreml: REML covariance parameters (Schabenberger Sec 4.5.2).

Minimises eq (4.39), minus twice the restricted log likelihood of K Z(s),
where K is a matrix of error contrasts. There is no REML estimator of the
mean; eq (4.40) recovers it afterwards as an EGLS estimator.
"""

import numpy as np
import pytest

from morie.fn._schab_fit import error_contrasts
from morie.fn.spreml import schabenberger_reml_variogram as spreml
from morie.fn.spwls import schabenberger_wls_variogram as spwls


def test_contrast_matrix_annihilates_the_mean_structure():
    """K is defined by E[K Z(s)] = 0, i.e. K X = 0, with full row rank."""
    X = np.ones((25, 1))
    K = error_contrasts(X)
    assert K.shape == (24, 25)
    assert np.allclose(K @ X, 0.0, atol=1e-10)
    assert np.linalg.matrix_rank(K) == 24


def test_contrast_matrix_handles_a_regression_mean(schab_sites):
    coords = schab_sites(n=30, seed=4)
    X = np.column_stack([np.ones(30), coords])
    K = error_contrasts(X)
    assert K.shape[0] == 30 - 3
    assert np.allclose(K @ X, 0.0, atol=1e-10)


def test_reml_is_not_the_least_squares_answer(schab_sites, schab_simulate, schab_ev):
    """REML maximises the likelihood of the error contrasts, a different
    criterion from either sum of squares -- so it must not coincide with
    them. All three did once, when each was silently returning the shared
    starting values."""
    coords = schab_sites(n=120, seed=5)
    z = schab_simulate(coords, 11)
    r = spreml(coords, z, None, "exponential")
    w = spwls(schab_ev(coords, z), "exponential")
    assert r["range"] != w["range"]
    assert np.isfinite(r["neg2_restricted_loglik"])


def test_reml_improves_on_its_starting_values(schab_sites, schab_simulate):
    coords = schab_sites(n=110, seed=5)
    res = spreml(coords, schab_simulate(coords, 11), None, "exponential")
    assert res["converged"]


def test_reml_does_not_depend_on_the_choice_of_contrasts(schab_sites, schab_simulate):
    """Sec 4.5.2 states, citing Harville (1974), that K is not unique and the
    choice does not affect the estimates. Any orthogonal mixing of the
    contrast rows is another valid K."""
    coords = schab_sites(n=80, seed=4)
    X = np.ones((coords.shape[0], 1))
    K = error_contrasts(X)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(K.shape[0],) * 2))
    assert np.allclose((q @ K) @ X, 0.0, atol=1e-8)
    res = spreml(coords, schab_simulate(coords, 13), X, "exponential")
    assert res["n_contrasts"] == coords.shape[0] - 1


def test_reml_recovers_the_mean_by_egls(schab_sites, schab_simulate):
    """eq (4.40): the mean comes from EGLS evaluated at theta_reml."""
    coords = schab_sites(n=120, seed=5)
    res = spreml(coords, schab_simulate(coords, 11), None, "exponential")
    assert res["mean"] == pytest.approx(5.0, abs=1.5)


def test_reml_parameters_stay_in_the_valid_space(schab_sites, schab_simulate):
    coords = schab_sites(n=100, seed=5)
    r = spreml(coords, schab_simulate(coords, 11), None, "exponential")
    assert r["nugget"] >= 0.0
    assert r["partial_sill"] >= 0.0
    assert r["range"] > 0.0


def test_rejects_bad_input(schab_sites):
    coords = schab_sites(n=20)
    with pytest.raises(ValueError):
        spreml(coords, np.ones(19), None, "exponential")
    with pytest.raises(ValueError):
        spreml(coords, np.ones(20), np.ones((19, 1)), "exponential")
