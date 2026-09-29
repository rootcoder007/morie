"""Tests for misckit: Crime Severity Index, U-learner, repeated k-fold, FastICA, rStress MDS."""

import math

from morie.fn.misckit import (
    crime_severity_index,
    csi_weights,
    fast_ica,
    repeated_kfold_indices,
    rstress_mds,
    u_learner_cate,
)


def test_csi_is_base_normalised_weighted_rate():
    w = csi_weights([0.5, 0.9, 0.1], [30, 400, 10])
    r = crime_severity_index([[10, 2, 5], [8, 3, 7], [12, 1, 4]], w, [1000, 1100, 1050], base=1)
    rates = [
        (10 * w[0] + 2 * w[1] + 5 * w[2]) / 1000,
        (8 * w[0] + 3 * w[1] + 7 * w[2]) / 1100,
        (12 * w[0] + w[1] + 4 * w[2]) / 1050,
    ]
    assert r.index[1] == 100.0
    assert abs(r.index[0] - 100 * rates[0] / rates[1]) <= 1e-12


def test_u_learner_recovers_linear_effect():
    X = [[math.sin(i * 0.37), math.cos(i * 0.91)] for i in range(80)]
    d = [1 if math.sin(i * 1.7) > 0 else 0 for i in range(80)]
    y = [(1.5 + 0.8 * X[i][0]) * d[i] + X[i][1] for i in range(80)]
    r = u_learner_cate(y, d, X, l2=1e-8)
    assert abs(r.coefficients[0] - 1.5) < 0.2 and abs(r.coefficients[1] - 0.8) < 0.3


def test_repeated_kfold_balance():
    for f in repeated_kfold_indices(23, 5, 4, seed=1):
        counts = [f.count(k) for k in range(5)]
        assert max(counts) - min(counts) <= 1 and sum(counts) == 23


def test_fast_ica_unmixes_and_sources_are_white():
    k = range(200)
    S = [[math.sin(i * 0.13), ((i * 7) % 11) / 5.0 - 1.0, math.cos(i * 0.029) ** 3] for i in k]
    X = [[s[0] + 0.5 * s[1] + 0.2 * s[2], 0.3 * s[0] - s[1] + 0.4 * s[2], 0.6 * s[0] + 0.2 * s[1] - s[2]] for s in S]
    r = fast_ica(X, 3, tol=1e-10, max_iter=500)
    n = len(X)
    for a in range(3):
        for b in range(3):
            c = sum(r.S[t][a] * r.S[t][b] for t in range(n)) / n
            assert abs(c - (a == b)) <= 1e-8
    for j in range(3):
        src = [s[j] for s in S]
        ms = sum(src) / n
        sd = math.sqrt(sum((v - ms) ** 2 for v in src) / n)
        best = max(abs(sum((src[t] - ms) * r.S[t][q] for t in range(n)) / (n * sd)) for q in range(3))
        assert best > 0.95


def test_rstress_exact_configuration():
    P = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.5, 1.0)]
    D = [[math.dist(a, b) for b in P] for a in P]
    r1 = rstress_mds(D, r=1.0)
    assert r1.stress <= 1e-6
    Dh = [[v**0.5 for v in row] for row in D]
    rh = rstress_mds(Dh, r=0.5)
    assert rh.stress <= 1e-6
