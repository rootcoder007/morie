"""stblocks: moments, decompositions and filters recomputed from their defining formulas."""

import math

import pytest

from morie.fn._qpcore import inverse, solve
from morie.fn._rng import random_normal
from morie.fn._sci_core import digamma
from morie.fn.stblocks import (
    arma_acf,
    carroll_st_correlation,
    harmonic_ozone_dlm,
    normal_gamma_dlm,
    partitioned_covariance,
    process_convolution_covariance,
    separable_prewhiten,
    wishart_moments,
)


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def test_wishart_one_dimensional_is_scaled_chi_square():
    s2, d = 1.7, 6.5
    w = wishart_moments([[s2]], d)
    assert w.mean[0][0] == pytest.approx(d * s2, rel=1e-14)
    assert w.mean_inverse[0][0] == pytest.approx(1 / (s2 * (d - 2)), rel=1e-14)
    assert w.mean_logdet == pytest.approx(math.log(2 * s2) + digamma(d / 2), rel=1e-13)
    iw = wishart_moments([[s2]], d, inverse_wishart=True)
    assert iw.mean[0][0] == pytest.approx(s2 / (d - 2), rel=1e-14)
    assert iw.mean_logdet == pytest.approx(math.log(s2 / 2) - digamma(d / 2), rel=1e-13)
    with pytest.raises(ValueError):
        wishart_moments([[1.0, 0.0], [0.0, 1.0]], 0.5)


def test_partitioned_covariance_reconstructs_and_conditions():
    S = [[3.0, 1.0, 0.5], [1.0, 2.0, 0.4], [0.5, 0.4, 1.0]]
    r = partitioned_covariance(S, 1, x2=[1.0, -2.0], mean=[0.5, 0.0, 1.0])
    for i in range(3):
        for j in range(3):
            assert r.reconstructed[i][j] == pytest.approx(S[i][j], abs=1e-12)
    S22 = [[2.0, 0.4], [0.4, 1.0]]
    tau = solve(S22, [1.0, 0.5])
    assert r.tau[0] == pytest.approx(tau, rel=1e-12)
    assert r.sigma_cond[0][0] == pytest.approx(3.0 - tau[0] * 1.0 - tau[1] * 0.5, rel=1e-12)
    assert r.cond_mean[0] == pytest.approx(0.5 + tau[0] * 1.0 + tau[1] * (-3.0), rel=1e-12)


def test_prewhitening_gives_identity_temporal_correlation():
    R = [[0.7 ** abs(i - j) for j in range(4)] for i in range(4)]
    r = separable_prewhiten([[1.0, 2.0, 0.5, -1.0]], R)
    M = r.rho_t_inv_sqrt
    I4 = _mm(_mm(M, R), M)
    for i in range(4):
        for j in range(4):
            assert I4[i][j] == pytest.approx(float(i == j), abs=1e-12)


def test_normal_gamma_with_known_variance_is_the_kalman_filter():
    y = random_normal(15, seed=3)
    V, Ws = 0.6, 0.3
    r = normal_gamma_dlm(y, [1.0], [[1.0]], [[Ws]], [0.0], [[2.0]], 1e12, V)
    m, C = 0.0, 2.0 * V
    for t, yt in enumerate(y):
        R = C + V * Ws
        K = R / (R + V)
        m, C = m + K * (yt - m), R - K * R
        assert r.m[t][0] == pytest.approx(m, rel=1e-9, abs=1e-12)
        assert r.C[t][0][0] == pytest.approx(C, rel=1e-9)
    # the variance update recursion
    s = normal_gamma_dlm(y[:3], [1.0], [[1.0]], [[0.5]], [0.0], [[1.0]], 2.0, 1.5)
    nS = 2.0 * 1.5
    for t in range(3):
        nS += (y[t] - s.f[t]) ** 2 / s.Q[t]
        assert s.S[t] == pytest.approx(nS / (3.0 + t), rel=1e-12)


def _logdet(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(v) if i == j else v / L[j][j]
    return 2 * sum(math.log(L[i][i]) for i in range(n))


def test_ozone_dlm_likelihood_equals_joint_gaussian():
    e = random_normal(12, seed=8)
    coords = [[0.0, 0.0], [1.0, 1.0]]
    a, lam, s2, w, c0 = [0.4, -0.1], 1.2, 0.3, [0.1, 0.05, 0.02], 2.0
    times = [0, 1, 2, 3, 4, 5]
    Y = [[1.0 + e[2 * t + s] for s in range(2)] for t in range(6)]
    r = harmonic_ozone_dlm(Y, times, coords, a, lam, s2, w, c0=c0)
    Wd = [w[0], w[1], w[1], w[2], w[2]]
    Fs = []
    for t in times:
        s1 = math.cos(math.pi * t / 12) + a[0] * math.sin(math.pi * t / 12)
        s2t = math.cos(math.pi * t / 6) + a[1] * math.sin(math.pi * t / 6)
        Fs.append([[1.0, s1, 0.0, s2t, 0.0], [1.0, 0.0, s1, 0.0, s2t]])
    V = [[s2, s2 * math.exp(-math.sqrt(2) / lam)], [s2 * math.exp(-math.sqrt(2) / lam), s2]]
    n = 12
    Sig = [[0.0] * n for _ in range(n)]
    for t in range(6):
        for u in range(6):
            Ct = [[(c0 + (min(t, u) + 1) * Wd[i]) if i == j else 0.0 for j in range(5)] for i in range(5)]
            B = _mm(_mm(Fs[t], Ct), [list(c) for c in zip(*Fs[u])])
            for i in range(2):
                for j in range(2):
                    Sig[2 * t + i][2 * u + j] = B[i][j] + (V[i][j] if t == u else 0.0)
    yv = [v for row in Y for v in row]
    q = sum(a_ * b for a_, b in zip(yv, solve(Sig, yv)))
    ll = -0.5 * (n * math.log(2 * math.pi) + _logdet(Sig) + q)
    assert r.loglik == pytest.approx(ll, rel=1e-9)
    assert len(inverse(Sig)) == n


def test_carroll_and_process_convolution_closed_forms():
    r = carroll_st_correlation([[0, 0], [2, 0]], [0, 1], [-0.3, 0.1, 0.02], [-0.1, -0.2, 0.0])
    rho = math.exp(-0.1 - 0.2) * math.exp(-0.3 + 0.1 + 0.02) ** 2
    assert r.correlation[0][1] == pytest.approx(rho, rel=1e-13)
    assert r.min_eigenvalue == pytest.approx(1 - rho, rel=1e-12)
    pts = [[0.0, 0.0], [1.0, 0.5], [0.3, -1.0]]
    c = process_convolution_covariance(pts, [[1.0, 0.0], [0.0, 1.0]], normalize=True).covariance
    for i in range(3):
        for j in range(3):
            h2 = sum((pts[i][k] - pts[j][k]) ** 2 for k in range(2))
            assert c[i][j] == pytest.approx(math.exp(-h2 / 4), rel=1e-12)
    raw = process_convolution_covariance(pts, [[0.5, 0.1], [0.1, 0.8]], sigma2=3.0).covariance
    det = (2 * 0.5) * (2 * 0.8) - (2 * 0.1) ** 2
    assert raw[1][1] == pytest.approx(3.0 / (2 * math.pi) / math.sqrt(det), rel=1e-12)


def test_arma_acf_closed_forms():
    th = 0.6
    assert arma_acf(ma=[th], lag_max=3) == pytest.approx([1.0, th / (1 + th * th), 0.0, 0.0], abs=1e-15)
    assert arma_acf(ar=[0.8], lag_max=5) == pytest.approx([0.8**k for k in range(6)], rel=1e-13)
    p1, p2 = 0.5, 0.3
    r = arma_acf(ar=[p1, p2], lag_max=6)
    ref = [1.0, p1 / (1 - p2)]
    for _ in range(5):
        ref.append(p1 * ref[-1] + p2 * ref[-2])
    assert r == pytest.approx(ref, rel=1e-12)
    with pytest.raises(ValueError):
        arma_acf()
