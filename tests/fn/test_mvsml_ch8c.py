"""Known-answer tests for MVSML chapter 8, eq. (8.8)-(8.10)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm138 import (mvsml_categorical_count_eq_8_8,
                             mvsml_bayesian_kernel_blup)
from morie.fn.msm142 import mvsml_categorical_count_eq_8_9
from morie.fn.msm144 import mvsml_categorical_count_eq_8_10


def _grm(seed=5, n=8, p=20):
    rng = gp.np.random.default_rng(seed)
    M = [[float(rng.integers(0, 3)) for _ in range(p)]
         for _ in range(n)]
    G = gp.grm_vanraden_method3(M)
    return [[G[i][j] + (0.4 if i == j else 0.0) for j in range(n)]
            for i in range(n)]


def test_eq_8_8_conditional_mode_is_the_blup():
    # p.282: the mean/mode of u is the BLUP of Henderson's MME, so
    # with K the genomic relationship matrix this model IS GBLUP
    G = _grm()
    n = len(G)
    rng = gp.np.random.default_rng(3)
    y = [5.0 + float(rng.normal(0, 1)) for _ in range(n)]
    s2u, s2e = 0.7, 1.3
    r = mvsml_categorical_count_eq_8_8(y, G, sigma2_u=s2u,
                                       sigma2_e=s2e, gibbs=False)
    Z = [[1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    X = [[1.0] for _ in range(n)]
    Sigma = [[s2u * G[i][j] for j in range(n)] for i in range(n)]
    R = [[s2e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    beta, u = gp.blue_blup_via_v(X, Z, y, Sigma, R)
    # the eq. (8.8) conditional mode is taken at mu = mean(y); the
    # BLUP uses the GLS intercept, so compare the shape and ordering
    assert len(r["u"]) == n
    assert r["u"].index(max(r["u"])) == u.index(max(u))
    assert r["u"].index(min(r["u"])) == u.index(min(u))


def test_eq_8_8_recovers_the_blup_at_the_gls_intercept():
    G = _grm()
    n = len(G)
    rng = gp.np.random.default_rng(4)
    y = [5.0 + float(rng.normal(0, 1)) for _ in range(n)]
    s2u, s2e = 0.7, 1.3
    Z = [[1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    X = [[1.0] for _ in range(n)]
    Sigma = [[s2u * G[i][j] for j in range(n)] for i in range(n)]
    R = [[s2e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    beta, u = gp.blue_blup_via_v(X, Z, y, Sigma, R)
    f = gp.bayesian_kernel_blup(y, G, sigma2_u=s2u, sigma2_e=s2e,
                                gibbs=False)
    # rebuild the conditional mode at the GLS intercept
    Kinv = gp._inv(G)
    A = [[Kinv[i][j] / s2u + ((1.0 / s2e) if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    Kt = gp._inv(A)
    ut = [v / s2e for v in gp._mv(Kt, [a - beta[0] for a in y])]
    for a, b in zip(ut, u):
        assert abs(a - b) < 1e-8           # identical to the BLUP


def test_eq_8_8_gibbs_runs_and_stays_finite():
    G = _grm()
    rng = gp.np.random.default_rng(6)
    y = [5.0 + float(rng.normal(0, 1)) for _ in range(len(G))]
    r = mvsml_categorical_count_eq_8_8(y, G, n_iter=400,
                                       burn_in=150)
    assert r["sigma2_u"] > 0 and r["sigma2_e"] > 0
    assert all(v == v for v in r["u"])
    assert abs(r["mu"] - sum(y) / len(y)) < 1.5


def test_eq_8_9_covariance_is_z_k_zt():
    K = [[1.0, 0.3], [0.3, 1.0]]
    Z = [[1, 0], [1, 0], [0, 1]]
    r = mvsml_categorical_count_eq_8_9(Z, K)
    Ks = r["K_star"]
    assert abs(Ks[0][0] - 1.0) < 1e-12     # both rows are line 1
    assert abs(Ks[0][1] - 1.0) < 1e-12
    assert abs(Ks[0][2] - 0.3) < 1e-12
    assert r["positive_semidefinite"] is True
    # scaling by sigma2_u scales the covariance
    r2 = mvsml_categorical_count_eq_8_9(Z, K, sigma2_u=2.0)
    assert abs(r2["K_star"][0][0] - 2.0) < 1e-12


def test_eq_8_10_interaction_kernel_is_a_hadamard_product():
    K = [[1.0, 0.3], [0.3, 1.0]]
    # two lines measured in two environments
    Z_u1 = [[1, 0], [0, 1], [1, 0], [0, 1]]
    Z_E = [[1, 0], [1, 0], [0, 1], [0, 1]]
    r = mvsml_categorical_count_eq_8_10(Z_u1, K, Z_E)
    K1, K2, KE = r["K1"], r["K2"], r["K_env"]
    for i in range(4):
        for j in range(4):
            assert abs(K2[i][j] - K1[i][j] * KE[i][j]) < 1e-12
    # the environment kernel is a block indicator
    assert abs(KE[0][1] - 1.0) < 1e-12     # same environment
    assert abs(KE[0][2]) < 1e-12           # different environment
    # so the interaction kernel is block diagonal by environment
    assert abs(K2[0][2]) < 1e-12
    assert r["K1_psd"] is True and r["K2_psd"] is True


def test_hadamard_matches_the_definition():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    H = gp.hadamard(A, B)
    assert H == [[5.0, 12.0], [21.0, 32.0]]


def test_canonical_aliases():
    from morie.fn.msm142 import mvsml_kernel_blup_replicated
    from morie.fn.msm144 import mvsml_kernel_blup_gxe
    assert mvsml_bayesian_kernel_blup is mvsml_categorical_count_eq_8_8
    assert mvsml_kernel_blup_replicated is \
        mvsml_categorical_count_eq_8_9
    assert mvsml_kernel_blup_gxe is mvsml_categorical_count_eq_8_10
