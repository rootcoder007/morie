"""Verification tests for msm055.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 6, eq. 6.5 p.177, replicated individuals. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5


Y = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5]
Z = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
     [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
G = [[1.0, 0.25], [0.25, 1.0]]


def test_the_predictor_covariance_is_the_incidence_sandwich():
    # p.177: K_L = Var(Z g) = Z G Z', which BGLR is given in place of
    # the predictor it cannot take directly
    res = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=200, burn_in=50, seed=7)
    K = res["K_L"]
    for i in range(6):
        for j in range(6):
            expected = sum(Z[i][a] * G[a][b] * Z[j][b]
                           for a in range(2) for b in range(2))
            assert K[i][j] == pytest.approx(expected, rel=1e-12)


def test_the_predictor_covariance_is_symmetric():
    res = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=200, burn_in=50, seed=7)
    K = res["K_L"]
    for i in range(6):
        for j in range(6):
            assert K[i][j] == pytest.approx(K[j][i], rel=1e-12)


def test_replicates_of_one_genotype_receive_the_same_effect():
    # rows 0, 1 and 4 carry genotype 1 and rows 2, 3 and 5 genotype 2,
    # so eq 6.5 gives each replicate of a genotype the same g
    res = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=400, burn_in=100, seed=7)
    g = list(res["g"])
    assert abs(g[0] - g[1]) < 1e-3
    assert abs(g[0] - g[4]) < 1e-3
    assert abs(g[2] - g[3]) < 1e-3
    assert abs(g[2] - g[5]) < 1e-3
    assert abs(g[0] - g[2]) > 1e-3


def test_an_identity_relationship_gives_an_identity_covariance():
    n = 3
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    res = mvsml_bayesian_regression_eq_6_5([1.0, 2.0, 3.0], I, I, n_iter=200, burn_in=50, seed=3)
    for i in range(n):
        for j in range(n):
            assert res["K_L"][i][j] == pytest.approx(I[i][j], abs=1e-12)


def test_the_same_seed_reproduces_the_same_chain():
    a = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=200, burn_in=50, seed=11)
    b = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=200, burn_in=50, seed=11)
    assert a["estimate"] == b["estimate"]
    assert list(a["g"]) == list(b["g"])


def test_the_intercept_lands_inside_the_range_of_the_data():
    res = mvsml_bayesian_regression_eq_6_5(Y, Z, G, n_iter=400, burn_in=100, seed=7)
    assert min(Y) <= res["mu"] <= max(Y)
    assert res["sigma2"] > 0.0
