"""Known-answer tests for MVSML chapter 8, eq. (8.6)-(8.7)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm135 import (mvsml_categorical_count_eq_8_6,
                             mvsml_rkhs_mixed_equations)
from morie.fn.msm137 import mvsml_categorical_count_eq_8_7


def _setup(seed=7, n=9, p=20):
    rng = gp.np.random.default_rng(seed)
    X = [[float(rng.integers(0, 3)) for _ in range(p)]
         for _ in range(n)]
    K = gp.grm_vanraden_method3(X)
    K = [[K[i][j] + (0.5 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    C = [[1.0] for _ in range(n)]
    y = [5.0 + float(rng.normal(0, 1)) for _ in range(n)]
    return C, K, y, X


def test_eq_8_6_and_8_7_give_the_same_solution():
    # p.276: multiplying the second system of (8.6) by K^-1 gives
    # (8.7); "both parameterizations produce the same solution"
    C, K, y, _ = _setup()
    a = mvsml_categorical_count_eq_8_6(C, K, y, lam=0.7,
                                       sigma2_e=1.3)
    b = mvsml_categorical_count_eq_8_7(C, K, y, lam=0.7,
                                       sigma2_e=1.3)
    assert abs(a["theta"][0] - b["theta"][0]) < 1e-7
    for u, v in zip(a["beta"], b["beta"]):
        assert abs(u - v) < 1e-7
    for u, v in zip(a["fitted"], b["fitted"]):
        assert abs(u - v) < 1e-7


def test_reparameterization_two_sets_u_equal_to_k_beta():
    # p.276: y = C theta + u + e with u = K beta
    C, K, y, _ = _setup()
    r = mvsml_categorical_count_eq_8_6(C, K, y, lam=0.5)
    hand = gp._mv(K, r["beta"])
    for a, b in zip(r["u"], hand):
        assert abs(a - b) < 1e-9
    # fitted = C theta + u
    for i in range(len(y)):
        assert abs(r["fitted"][i]
                   - (r["theta"][0] + r["u"][i])) < 1e-9


def test_sigma2_beta_is_the_inverse_of_lambda():
    C, K, y, _ = _setup()
    r = mvsml_categorical_count_eq_8_6(C, K, y, lam=4.0)
    assert abs(r["sigma2_beta"] - 0.25) < 1e-12


def test_larger_lambda_shrinks_the_genomic_effects():
    C, K, y, _ = _setup()
    small = mvsml_categorical_count_eq_8_7(C, K, y, lam=0.01)
    large = mvsml_categorical_count_eq_8_7(C, K, y, lam=50.0)
    assert sum(abs(v) for v in large["u"]) < \
        sum(abs(v) for v in small["u"])
    # and the intercept absorbs the mean
    assert abs(large["theta"][0] - sum(y) / len(y)) < 0.3


def test_prediction_for_new_individuals_is_k_star_beta():
    # p.276: u_new = K_s beta, a single matrix-vector product
    C, K, y, X = _setup()
    n = len(K)
    rng = gp.np.random.default_rng(21)
    Xnew = [[float(rng.integers(0, 3)) for _ in range(len(X[0]))]
            for _ in range(3)]
    Ks = gp.kernel_matrix(Xnew, kernel="linear", Z=X)
    r = mvsml_categorical_count_eq_8_7(C, K, y, lam=0.5,
                                       K_star=Ks)
    assert len(r["u_new"]) == 3
    hand = gp._mv(Ks, r["beta"])
    for a, b in zip(r["u_new"], hand):
        assert abs(a - b) < 1e-12


def test_canonical_aliases():
    from morie.fn.msm137 import \
        mvsml_rkhs_mixed_equations_reduced
    assert mvsml_rkhs_mixed_equations is \
        mvsml_categorical_count_eq_8_6
    assert mvsml_rkhs_mixed_equations_reduced is \
        mvsml_categorical_count_eq_8_7
