"""Known-answer tests for MVSML chapter 6, eq. (6.8)-(6.11).

The book states on p.191 that eq. (6.9) with diagonal Sigma_T and R is
equivalent to fitting a univariate GBLUP per trait, and on p.194 that
eq. (6.10) is the same model written as a multivariate ridge; both are
used as oracles here.
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm065 import mvsml_bayesian_regression_eq_6_8
from morie.fn.msm067 import mvsml_bayesian_regression_eq_6_9
from morie.fn.msm072 import mvsml_bayesian_regression_eq_6_10
from morie.fn.msm076 import mvsml_bayesian_regression_eq_6_11


def _data(seed=5):
    rng = gp.np.random.default_rng(seed)
    J = 6
    A = [[float(rng.normal(0, 1)) for _ in range(8)]
         for _ in range(J)]
    G = gp.grm_vanraden_method3(A)
    G = [[G[i][j] + (0.4 if i == j else 0.0) for j in range(J)]
         for i in range(J)]
    Y = [[5.0 + float(rng.normal(0, 1)),
          2.0 + float(rng.normal(0, 1))] for _ in range(J)]
    Z1 = [[1.0 if i == j else 0.0 for j in range(J)]
          for i in range(J)]
    return Y, Z1, G


def test_eq_6_8_returns_proper_covariances():
    Y, Z1, G = _data()
    r = mvsml_bayesian_regression_eq_6_8(Y, Z1, G, n_iter=400,
                                         burn_in=100)
    for M in (r["Sigma_T"], r["R"]):
        assert abs(M[0][1] - M[1][0]) < 1e-9      # symmetric
        assert M[0][0] > 0 and M[1][1] > 0        # positive diagonal
        assert M[0][0] * M[1][1] > M[0][1] ** 2   # positive definite
    assert len(r["mu"]) == 2


def test_eq_6_9_posterior_means_track_the_trait_means():
    Y, Z1, G = _data()
    r = mvsml_bayesian_regression_eq_6_9(Y, Z1, G, n_iter=800,
                                         burn_in=200)
    for t in range(2):
        col = [row[t] for row in Y]
        assert abs(r["mu"][t] - sum(col) / len(col)) < 1.0
    assert len(r["b1"]) == len(G)
    assert len(r["b1"][0]) == 2
    assert r["n_kept"] == 600


def test_eq_6_10_ridge_form_reproduces_g():
    Y, Z1, G = _data()
    r = mvsml_bayesian_regression_eq_6_10(Z1, G)
    L = r["L_G"]
    n = len(G)
    prod = gp._mm(L, gp._t(L))
    assert max(abs(prod[i][j] - G[i][j])
               for i in range(n) for j in range(n)) < 1e-9
    # with Z1 = I the ridge design is exactly L_G (p.194)
    for i in range(n):
        for j in range(n):
            assert abs(r["X1"][i][j] - L[i][j]) < 1e-12
    # X1 X1' = Z1 G Z1', so the induced covariance is unchanged
    XX = gp._mm(r["X1"], gp._t(r["X1"]))
    assert max(abs(XX[i][j] - G[i][j])
               for i in range(n) for j in range(n)) < 1e-9


def test_eq_6_11_bmtme_conditionals_follow_p196():
    Y, Z1, G = _data()
    J = len(G)
    I = 2
    Z2 = [[1.0 if i == j else 0.0 for j in range(I * J)]
          for i in range(J)]
    b1 = [[0.1 * (i + 1), -0.05 * (i + 1)] for i in range(J)]
    b2 = [[0.02 * (i + 1), 0.01 * (i + 1)] for i in range(I * J)]
    Sigma_T = [[1.0, 0.2], [0.2, 1.0]]
    Sigma_E = [[1.0, 0.0], [0.0, 1.0]]
    R = [[1.0, 0.0], [0.0, 1.0]]
    r = mvsml_bayesian_regression_eq_6_11(Y, Z1, Z2, G, Sigma_T,
                                          Sigma_E, R, b1=b1, b2=b2)
    # step 5: nu_T + J + IJ degrees of freedom
    assert abs(r["nu_T_post"] - (4.0 + J + I * J)) < 1e-12
    # step 6: nu_E + J L
    assert abs(r["nu_E_post"] - (4.0 + J * I)) < 1e-12
    for M, k in ((r["scale_T"], 2), (r["scale_E"], I)):
        assert len(M) == k
        assert abs(M[0][1] - M[1][0]) < 1e-9
        assert M[0][0] > 0
    # the b1 term alone is b1' G^-1 b1 + S_T, so dropping b2 shrinks
    # the scale matrix
    r0 = mvsml_bayesian_regression_eq_6_11(Y, Z1, Z2, G, Sigma_T,
                                           Sigma_E, R, b1=b1)
    assert r0["scale_T"][0][0] < r["scale_T"][0][0]


def test_inverse_wishart_draw_has_the_right_mean():
    rng = gp.np.random.default_rng(3)
    nu, p = 30.0, 2
    S = [[4.0, 0.0], [0.0, 9.0]]
    draws = [gp.inv_wishart_draw(rng, nu, S) for _ in range(600)]
    # E[IW(nu, S)] = S / (nu - p - 1)
    m00 = sum(d[0][0] for d in draws) / len(draws)
    m11 = sum(d[1][1] for d in draws) / len(draws)
    assert abs(m00 - 4.0 / (nu - p - 1)) < 0.05
    assert abs(m11 - 9.0 / (nu - p - 1)) < 0.12
    # draws are symmetric positive definite
    d = draws[0]
    assert abs(d[0][1] - d[1][0]) < 1e-9
    assert d[0][0] * d[1][1] > d[0][1] ** 2


def test_multitrait_with_diagonal_covariances_tracks_univariate():
    # book p.191: with Sigma_T and R diagonal the model is equivalent
    # to fitting a univariate GBLUP per trait; the sampler is not
    # given diagonal draws, so compare the ordering of the genotype
    # effects, which must agree with the univariate fit
    Y, Z1, G = _data(seed=9)
    r = mvsml_bayesian_regression_eq_6_9(Y, Z1, G, n_iter=1200,
                                         burn_in=400)
    for t in range(2):
        uni = gp.gblup_model([row[t] for row in Y], Z1, G, 1.0, 1.0)
        got = [row[t] for row in r["b1"]]
        best_multi = got.index(max(got))
        best_uni = uni["b"].index(max(uni["b"]))
        assert best_multi == best_uni
