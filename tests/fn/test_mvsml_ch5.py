"""Known-answer tests for MVSML chapter 5 (linear mixed models).

Checks follow the book's own statements: the MME and V-based solutions
agree (p.36/p.147), REML differs from ML by the -1/2 log|X'V^-1X| term
(p.146), the EM updates of pp.143-144 converge to the ML variance
components, and the multi-trait models of eq. (5.5)/(5.6) reduce to
separate univariate GBLUP fits when the covariance matrices are
diagonal (pp.153, 155).
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm010 import mvsml_linear_mixed_models_eq_5_1
from morie.fn.msm011 import mvsml_linear_mixed_models_eq_5_2
from morie.fn.msm015 import mvsml_linear_mixed_models_eq_5_3
from morie.fn.msm018 import mvsml_linear_mixed_models_eq_5_4
from morie.fn.msm026 import mvsml_linear_mixed_models_eq_5_5
from morie.fn.msm028 import mvsml_linear_mixed_models_eq_5_5a
from morie.fn.msm032 import mvsml_linear_mixed_models_eq_5_6

X6 = [[1.0]] * 6
Z6 = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
      [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
Y6 = [5.0, 5.2, 4.8, 6.4, 6.6, 6.2]
D2 = [[0.5, 0.0], [0.0, 0.5]]
I2 = [[1.0, 0.0], [0.0, 1.0]]


def test_eq_5_1_marginal_variance_and_blup():
    r = mvsml_linear_mixed_models_eq_5_1(X6, Z6, Y6, D2)
    # Var(Y) = Z D Z' + R: within a level 0.5 + 1, across 0.5, zero
    # between levels
    V = r["V"]
    assert abs(V[0][0] - 1.5) < 1e-12
    assert abs(V[0][1] - 0.5) < 1e-12
    assert abs(V[0][3]) < 1e-12
    # BLUE is between the two level means, BLUPs shrink toward zero
    assert 5.0 < r["beta"][0] < 6.4
    assert abs(r["blup"][0]) < abs(5.0 - r["beta"][0])
    assert abs(sum(r["blup"])) < 1e-9


def test_eq_5_1_matches_the_mme_solution():
    Sigma_inv = [[2.0, 0.0], [0.0, 2.0]]
    beta, b = gp.mme_solve(X6, Z6, Y6, Sigma_inv)
    r = mvsml_linear_mixed_models_eq_5_1(X6, Z6, Y6, D2)
    assert abs(beta[0] - r["beta"][0]) < 1e-9
    for a, c in zip(b, r["blup"]):
        assert abs(a - c) < 1e-9


def test_eq_5_2_likelihood_and_reml_relation():
    ml = mvsml_linear_mixed_models_eq_5_2(X6, Z6, Y6, D2)
    re = mvsml_linear_mixed_models_eq_5_2(X6, Z6, Y6, D2,
                                          restricted=True)
    # REML = ML - 1/2 log|X'V^-1X| (p.146)
    V = gp.lmm_marginal_v(Z6, D2)
    Vi = gp._inv(V)
    A = gp._mm(gp._mm(gp._t(X6), Vi), X6)
    assert abs(re["loglik"] - (ml["loglik"]
                               + 0.5 * len(Y6)
                               * math.log(2.0 * math.pi)
                               - 0.5 * gp._logdet(A))) < 1e-9
    # the likelihood is maximized at the GLS beta
    worse = mvsml_linear_mixed_models_eq_5_2(X6, Z6, Y6, D2,
                                             beta=[0.0])
    assert worse["loglik"] < ml["loglik"]


def test_em_recovers_the_ml_variance_components():
    r = gp.em_lmm(X6, Z6, Y6, n_iter=800)
    # grand mean, and level means 5.0 / 6.4 give ML sigma2_b ~ 0.49
    assert abs(r["beta"][0] - 5.7) < 1e-6
    assert abs(r["D"][0][0] - 0.4847) < 5e-3
    # residual ML variance is the within-level ML variance 0.0267
    assert 0.02 < r["sigma2"] < 0.05
    # the EM likelihood never decreases
    ll_prev = None
    for it in (1, 5, 20, 200):
        f = gp.em_lmm(X6, Z6, Y6, n_iter=it)
        ll, _ = gp.lmm_loglik(X6, Z6, Y6, f["D"], R=[
            [f["sigma2"] if i == j else 0.0 for j in range(6)]
            for i in range(6)])
        if ll_prev is not None:
            assert ll >= ll_prev - 1e-6
        ll_prev = ll


def test_eq_5_3_gblup_uses_the_relationship_matrix():
    # three lines, two replicates each
    Z = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
         [0, 0, 1], [0, 0, 1]]
    y = [5.1, 4.9, 6.0, 6.2, 5.5, 5.7]
    G = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
    r = mvsml_linear_mixed_models_eq_5_3(y, Z, G, sigma2_g=0.5)
    assert abs(r["mu"] - sum(y) / len(y)) < 0.2
    # line 2 has the highest phenotype so it gets the highest GEBV
    assert r["gebv"][1] == max(r["gebv"])
    # related lines 1 and 2 borrow from each other: with an identity
    # relationship matrix line 1's GEBV is lower
    Gi = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    r2 = mvsml_linear_mixed_models_eq_5_3(y, Z, Gi, sigma2_g=0.5)
    assert r["gebv"][0] > r2["gebv"][0]


def test_eq_5_4_gxe_splits_line_and_interaction_effects():
    # two environments x two lines, one replicate each
    y = [5.0, 6.0, 5.4, 6.8]
    X_E = [[0.0], [0.0], [1.0], [1.0]]        # environment 2 dummy
    Z_L = [[1, 0], [0, 1], [1, 0], [0, 1]]
    Z_EL = [[1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]]
    G = [[1.0, 0.0], [0.0, 1.0]]
    Sigma_E = [[0.3, 0.0], [0.0, 0.3]]
    r = mvsml_linear_mixed_models_eq_5_4(y, X_E, Z_L, Z_EL, G,
                                         sigma2_g=0.5,
                                         Sigma_E=Sigma_E)
    assert len(r["b_lines"]) == 2
    assert len(r["b_gxe"]) == 4
    # line 2 outperforms line 1 in both environments
    assert r["b_lines"][1] > r["b_lines"][0]
    # environment 2 has the higher mean, so its fixed effect is > 0
    assert r["beta"][1] > 0


def test_eq_5_5_reduces_to_univariate_when_diagonal():
    # book p.153: with diagonal Sigma_T and R the multi-trait model is
    # equivalent to fitting each trait separately
    Y = [[5.0, 2.0], [6.0, 3.0], [5.5, 2.4]]
    Z = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    G = [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]
    Sigma_T = [[0.4, 0.0], [0.0, 0.9]]
    R_T = [[1.0, 0.0], [0.0, 2.0]]
    r = mvsml_linear_mixed_models_eq_5_5(Y, Z, G, Sigma_T, R_T)
    for t, (s2g, s2e) in enumerate([(0.4, 1.0), (0.9, 2.0)]):
        uni = gp.gblup_model([row[t] for row in Y], Z, G, s2g, s2e)
        assert abs(r["mu"][t] - uni["mu"]) < 1e-8
        got = [blk[t] for blk in r["b_by_line"]]
        for a, b in zip(got, uni["b"]):
            assert abs(a - b) < 1e-8


def test_eq_5_5_correlated_traits_differ_from_separate_fits():
    Y = [[5.0, 2.0], [6.0, 3.0], [5.5, 2.4]]
    Z = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    G = [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]
    diag = mvsml_linear_mixed_models_eq_5_5(
        Y, Z, G, [[0.4, 0.0], [0.0, 0.9]], [[1.0, 0.0], [0.0, 2.0]])
    corr = mvsml_linear_mixed_models_eq_5_5(
        Y, Z, G, [[0.4, 0.3], [0.3, 0.9]], [[1.0, 0.0], [0.0, 2.0]])
    assert abs(corr["b"][0] - diag["b"][0]) > 1e-6


def test_eq_5_5a_adds_fixed_effects():
    Y = [[5.0, 2.0], [6.0, 3.0], [5.5, 2.4]]
    Z = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    G = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    Sigma_T = [[0.4, 0.0], [0.0, 0.9]]
    R_T = [[1.0, 0.0], [0.0, 2.0]]
    Xf = [[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]
    r = mvsml_linear_mixed_models_eq_5_5a(Y, Z, G, Sigma_T, R_T,
                                          X=Xf)
    plain = mvsml_linear_mixed_models_eq_5_5(Y, Z, G, Sigma_T, R_T)
    assert len(r["beta"]) == len(plain["mu"]) + 1
    assert abs(r["beta"][-1]) > 1e-9        # the covariate is used


def test_eq_5_6_reduces_to_univariate_when_all_diagonal():
    # book p.155: diagonal Sigma_T, Sigma_2T, Sigma_E and R make the
    # G x E multi-trait model equivalent to per-trait univariate fits
    Y = [[5.0, 2.0], [6.0, 3.0], [5.4, 2.2], [6.6, 3.4]]
    Z_L = [[1, 0], [0, 1], [1, 0], [0, 1]]
    Z_EL = [[1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]]
    G = [[1.0, 0.0], [0.0, 1.0]]
    Sigma_T = [[0.4, 0.0], [0.0, 0.9]]
    Sigma_2T = [[0.2, 0.0], [0.0, 0.5]]
    Sigma_E = [[1.0, 0.0], [0.0, 1.0]]
    R_T = [[1.0, 0.0], [0.0, 2.0]]
    r = mvsml_linear_mixed_models_eq_5_6(Y, Z_L, Z_EL, G, Sigma_T,
                                         Sigma_E, Sigma_2T, R_T)
    for t, (s2g, s2ge, s2e) in enumerate([(0.4, 0.2, 1.0),
                                          (0.9, 0.5, 2.0)]):
        yt = [row[t] for row in Y]
        # intercept only: an all-zero covariate column would make
        # X'V^-1X singular
        uni = gp.gxe_blup_model(yt, [[] for _ in range(4)], Z_L,
                                Z_EL, G, s2g,
                                [[s2ge, 0.0], [0.0, s2ge]], s2e)
        assert abs(r["b_lines"][t] - uni["b_lines"][0]) < 1e-7
        assert abs(r["b_lines"][2 + t] - uni["b_lines"][1]) < 1e-7


def test_kron_matches_the_definition():
    K = gp.kron([[1.0, 2.0], [3.0, 4.0]], [[0.0, 5.0], [6.0, 7.0]])
    assert K[0] == [0.0, 5.0, 0.0, 10.0]
    assert K[1] == [6.0, 7.0, 12.0, 14.0]
    assert K[3] == [18.0, 21.0, 24.0, 28.0]
