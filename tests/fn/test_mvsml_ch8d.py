"""Known-answer tests for MVSML chapter 8, eq. (8.11)-(8.12)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm145 import (mvsml_categorical_count_eq_8_11,
                             mvsml_kernel_eigen_design)
from morie.fn.msm152 import mvsml_categorical_count_eq_8_12


def _markers(seed=5, n=10, p=25):
    rng = gp.np.random.default_rng(seed)
    return [[float(rng.integers(0, 3)) for _ in range(p)]
            for _ in range(n)]


def test_eq_8_11_design_reproduces_the_kernel():
    # p.289: P = U S^(1/2) so that P P' = K, hence (8.8) and (8.11)
    # are equivalent models
    X = _markers()
    K = gp.kernel_matrix(X, kernel="linear")
    r = mvsml_categorical_count_eq_8_11(K)
    P = r["P"]
    n = len(K)
    PPt = gp._mm(P, gp._t(P))
    for i in range(n):
        for j in range(n):
            assert abs(PPt[i][j] - K[i][j]) < 1e-7
    assert r["rank"] <= n
    assert all(v > 0 for v in r["eigenvalues"])


def test_eq_8_11_rank_is_below_n_for_a_low_rank_kernel():
    # a linear kernel from p markers has rank at most p
    X = _markers(n=10, p=4)
    K = gp.kernel_matrix(X, kernel="linear")
    r = mvsml_categorical_count_eq_8_11(K)
    assert r["rank"] <= 4
    assert len(r["P"][0]) == r["rank"]


def test_nystrom_is_exact_when_all_lines_are_used():
    # Q = K_{n,m} K_{m,m}^-1 K_{n,m}' equals K when m = n
    X = _markers(n=8, p=20)
    K = gp.kernel_matrix(X, kernel="linear")
    p = len(X[0])
    K = [[v / p for v in row] for row in K]     # book's scaling
    ny = gp.nystrom_kernel(X, list(range(8)))
    for i in range(8):
        for j in range(8):
            assert abs(ny["Q"][i][j] - K[i][j]) < 1e-6


def test_nystrom_rank_is_m_and_q_is_psd():
    X = _markers(n=12, p=30)
    ny = gp.nystrom_kernel(X, [0, 3, 5, 7])
    assert ny["rank"] == 4
    ok, lam = gp.is_positive_semidefinite(ny["Q"])
    assert ok
    # Q has rank at most m, so at least n - m eigenvalues vanish
    assert sum(1 for v in lam if v > 1e-8) <= 4


def test_eq_8_12_design_reproduces_the_nystrom_kernel():
    # p.291: P = K_{n,m} U S^(-1/2), so P P' = Q
    X = _markers(n=12, p=30)
    r = mvsml_categorical_count_eq_8_12(X, [0, 3, 5, 7])
    P, Q = r["P"], r["Q"]
    PPt = gp._mm(P, gp._t(P))
    for i in range(12):
        for j in range(12):
            assert abs(PPt[i][j] - Q[i][j]) < 1e-6
    # only m effects are estimated
    assert len(P[0]) == r["rank"] <= 4


def test_eq_8_12_works_with_a_gaussian_kernel_too():
    # p.292: "the approximate kernel method can be used for any of
    # the kernels studied before"
    X = _markers(n=10, p=20)
    r = mvsml_categorical_count_eq_8_12(X, [1, 4, 6],
                                        kernel="gaussian")
    PPt = gp._mm(r["P"], gp._t(r["P"]))
    for i in range(10):
        for j in range(10):
            assert abs(PPt[i][j] - r["Q"][i][j]) < 1e-6


def test_more_landmarks_approximate_the_kernel_better():
    X = _markers(n=14, p=30)
    p = len(X[0])
    K = gp.kernel_matrix(X, kernel="linear")
    K = [[v / p for v in row] for row in K]

    def err(m_index):
        Q = gp.nystrom_kernel(X, m_index)["Q"]
        return max(abs(Q[i][j] - K[i][j])
                   for i in range(14) for j in range(14))

    assert err(list(range(10))) <= err(list(range(3))) + 1e-12


def test_canonical_aliases():
    from morie.fn.msm152 import mvsml_sparse_kernel_design
    assert mvsml_kernel_eigen_design is \
        mvsml_categorical_count_eq_8_11
    assert mvsml_sparse_kernel_design is \
        mvsml_categorical_count_eq_8_12
