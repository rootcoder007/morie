"""Tests for stdecomp: tensor decompositions, matrix factorisation, Haar wavelets, harmonic trends."""

import math

from morie.fn.stdecomp import (
    cp_als,
    haar_dwt,
    haar_dwt_2d,
    haar_mra,
    haar_mra_2d,
    harmonic_regression,
    matrix_factorization_als,
    tucker_hooi,
    wavelet_detrend,
)

a1, b1, c1 = [1.0, 2.0, -1.0, 0.5], [0.3, 1.0, 2.0], [1.0, -1.0, 0.5, 2.0, 1.5]
a2, b2, c2 = [0.5, -1.0, 1.0, 2.0], [1.0, 0.2, -0.7], [0.4, 1.0, 1.0, -0.5, 0.3]
T = [[[3 * a1[i] * b1[j] * c1[k] + a2[i] * b2[j] * c2[k] for k in range(5)] for j in range(3)] for i in range(4)]


def test_cp_recovers_an_exact_rank_two_tensor():
    r = cp_als(T, 2, max_iter=2000, tol=1e-15)
    for i in range(4):
        for j in range(3):
            for k in range(5):
                v = sum(r.weights[q] * r.A[i][q] * r.B[j][q] * r.C[k][q] for q in range(2))
                assert abs(v - T[i][j][k]) <= 1e-6
    assert r.weights[0] >= r.weights[1]
    assert abs(sum(v * v for v in [row[0] for row in r.A]) - 1) <= 1e-12


def test_tucker_full_rank_is_exact_and_orthonormal():
    r = tucker_hooi(T, (4, 3, 5), max_iter=2)
    for U in r.factors:
        n, q = len(U), len(U[0])
        for s in range(q):
            for t in range(q):
                assert abs(sum(U[i][s] * U[i][t] for i in range(n)) - (s == t)) <= 1e-10
    assert abs(r.fit - 1) <= 1e-6


def test_matrix_factorization_completes_low_rank_field():
    u, v = [1.0, 2.0, -1.0, 0.5, 3.0], [0.5, -1.0, 2.0, 1.0]
    X = [[u[i] * v[j] for j in range(4)] for i in range(5)]
    X[1][2] = None
    r = matrix_factorization_als(X, 1, l2=0.0, max_iter=5000, tol=1e-15)
    assert abs(r.fitted[1][2] - u[1] * v[2]) <= 1e-5


def test_haar_energy_and_exact_mra():
    x = [math.sin(i * 0.4) * 3 + 0.1 * i * i for i in range(16)]
    d = haar_dwt(x, 3)
    energy = sum(sum(w * w for w in lv) for lv in d.details) + sum(v * v for v in d.smooths[-1])
    assert abs(energy - sum(v * v for v in x)) <= 1e-9
    m = haar_mra(x, 3)
    for t in range(16):
        assert abs(sum(D[t] for D in m.details) + m.smooth[t] - x[t]) <= 1e-12
    w = wavelet_detrend(x, 2)
    assert max(abs(a + b - c) for a, b, c in zip(w.trend, w.detrended, x)) <= 1e-12


def test_haar_2d_energy_and_mra():
    F = [[math.sin(i * 0.3 + j * 0.8) + 0.05 * i * j for j in range(8)] for i in range(8)]
    d = haar_dwt_2d(F, 2)
    e = sum(v * v for row in d.approximation for v in row)
    e += sum(v * v for lv in d.levels for key in "hvd" for row in lv[key] for v in row)
    assert abs(e - sum(v * v for row in F for v in row)) <= 1e-10
    m = haar_mra_2d(F, 2)
    for i in range(8):
        for j in range(8):
            s = m.smooth[i][j] + sum(lv[k][i][j] for lv in m.details for k in "hvd")
            assert abs(s - F[i][j]) <= 1e-12


def test_harmonic_regression_recovers_coefficients():
    t = list(range(36))
    y = [2 + 0.03 * v + 1.5 * math.cos(2 * math.pi * v / 12 - 0.4) + 0.4 * math.sin(4 * math.pi * v / 12) for v in t]
    r = harmonic_regression(y, t, 12.0, 2)
    assert abs(r.coefficients[0] - 2) <= 1e-10 and abs(r.coefficients[1] - 0.03) <= 1e-12
    assert abs(r.amplitude[0] - 1.5) <= 1e-10 and abs(r.phase[0] - 0.4) <= 1e-10
    assert abs(r.amplitude[1] - 0.4) <= 1e-10 and abs(r.r2 - 1) <= 1e-12
