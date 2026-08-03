"""Known-answer tests for MVSML chapter 8, eq. (8.1)-(8.3)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm123 import (mvsml_categorical_count_eq_8_1,
                             mvsml_rkhs_objective)
from morie.fn.msm125 import mvsml_categorical_count_eq_8_2
from morie.fn.msm128 import mvsml_categorical_count_eq_8_3

X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
Y = [1.0, 2.0, 2.0, 3.0, 2.0]


def test_kernels_are_symmetric_and_psd():
    # p.255 properties 1 and 2
    for kern in ("linear", "gaussian", "polynomial", "exponential"):
        K = gp.kernel_matrix(X, kernel=kern)
        n = len(K)
        for i in range(n):
            for j in range(n):
                assert abs(K[i][j] - K[j][i]) < 1e-12
        ok, lam = gp.is_positive_semidefinite(K)
        assert ok, "%s kernel is not PSD: %s" % (kern, lam)


def test_linear_kernel_is_the_inner_product():
    # K(x_i, x_j) = phi(x_i)'phi(x_j) with phi the identity
    K = gp.kernel_matrix(X, kernel="linear")
    for i, a in enumerate(X):
        for j, b in enumerate(X):
            assert abs(K[i][j] - sum(u * w for u, w in zip(a, b))) \
                < 1e-12


def test_gaussian_kernel_hand_values():
    K = gp.kernel_matrix(X, kernel="gaussian", gamma=0.5)
    assert abs(K[0][0] - 1.0) < 1e-12          # zero distance
    d2 = 1.0                                    # between (0,0),(1,0)
    assert abs(K[0][1] - math.exp(-0.5 * d2)) < 1e-12
    # the Gaussian kernel decays with distance
    assert K[0][1] > K[0][3]


def test_representer_form_matches_the_definition():
    K = gp.kernel_matrix(X, kernel="linear")
    beta = [0.1, -0.2, 0.3, 0.0, 0.4]
    r = mvsml_categorical_count_eq_8_2(K, beta, eta0=0.5,
                                       K_train=K)
    for i in range(len(X)):
        hand = 0.5 + sum(beta[j] * K[i][j] for j in range(len(X)))
        assert abs(r["prediction"][i] - hand) < 1e-12
    # ||f||_H^2 = beta' K beta
    hand_norm = sum(beta[i] * K[i][j] * beta[j]
                    for i in range(len(X)) for j in range(len(X)))
    assert abs(r["rkhs_norm2"] - hand_norm) < 1e-12
    # the fit uses n coefficients, not p (p.253)
    assert r["n_coefficients"] == len(X)


def test_eq_8_3_fit_minimizes_the_eq_8_1_objective():
    K = gp.kernel_matrix(X, kernel="gaussian", gamma=0.5)
    lam = 0.1
    fit = mvsml_categorical_count_eq_8_3(K, Y, lam=lam)
    # the objective of (8.1) evaluated at the fitted values must not
    # be beaten by nearby perturbations
    base = mvsml_categorical_count_eq_8_1(K, Y, fit["beta"],
                                          eta0=fit["eta0"],
                                          lam=lam / 2.0)["objective"]
    for eps in (0.05, -0.05):
        pert = [b + eps for b in fit["beta"]]
        worse = mvsml_categorical_count_eq_8_1(
            K, Y, pert, eta0=fit["eta0"],
            lam=lam / 2.0)["objective"]
        assert worse >= base - 1e-9
    assert fit["penalty"] > 0


def test_larger_lambda_shrinks_the_rkhs_norm():
    K = gp.kernel_matrix(X, kernel="gaussian", gamma=0.5)
    small = mvsml_categorical_count_eq_8_3(K, Y, lam=0.01)
    large = mvsml_categorical_count_eq_8_3(K, Y, lam=10.0)
    assert gp.rkhs_norm(large["beta"], K) < \
        gp.rkhs_norm(small["beta"], K)


def test_generalized_kernel_model_links():
    K = gp.kernel_matrix(X, kernel="linear")
    beta = [0.1, -0.2, 0.3, 0.0, 0.4]
    ident = gp.generalized_kernel_model(K, beta, 0.0, "identity")
    logit = gp.generalized_kernel_model(K, beta, 0.0, "logit")
    logp = gp.generalized_kernel_model(K, beta, 0.0, "log")
    assert ident["mu"] == ident["eta"]
    assert all(0.0 < v < 1.0 for v in logit["mu"])
    assert all(v > 0.0 for v in logp["mu"])
    # the link is applied to the same linear predictor
    assert abs(logit["mu"][0]
               - 1.0 / (1.0 + math.exp(-ident["eta"][0]))) < 1e-12


def test_canonical_alias_is_the_same_function():
    from morie.fn.msm123 import (mvsml_categorical_count_eq_8_1,
                                 mvsml_rkhs_objective)
    assert mvsml_rkhs_objective is mvsml_categorical_count_eq_8_1


def test_non_psd_matrix_is_rejected():
    bad = [[1.0, 2.0], [2.0, 1.0]]        # eigenvalues 3 and -1
    ok, lam = gp.is_positive_semidefinite(bad)
    assert not ok
    assert min(lam) < 0
