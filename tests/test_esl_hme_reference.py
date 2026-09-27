"""Hierarchical mixture of experts (ESL 9.25-9.30): EM monotonicity and M-step fixed-point conditions."""

import math

from morie.fn.eslhme import esl_hme


def data():
    i = list(range(1, 61))
    X = [[((7 * t) % 59) / 59 * 4 - 2] for t in i]
    y = [(2 * r[0] if r[0] < 0 else -r[0] + 0.5) + 0.1 * math.cos(9 * t) for r, t in zip(X, i)]
    return X, y


def test_regression_hme():
    X, y = data()
    r = esl_hme(X, y, K=2, max_iter=500)
    p = r["loglik_path"]
    assert r["converged"] and all(b >= a - 1e-8 for a, b in zip(p, p[1:]))
    assert abs(r["loglik"] - 77.1569768184434) < 1e-6  # the R arm reaches the same value
    # E-step from the returned parameters, then each expert solves its weighted normal equations
    Z = [[1.0] + x for x in X]

    def lgate(coef, z):
        eta = [sum(c * v for c, v in zip(row, z)) for row in coef] + [0.0]
        m = max(eta)
        return [v - m - math.log(sum(math.exp(u - m) for u in eta)) for v in eta]

    H = []
    for z, yi in zip(Z, y):
        gt = lgate(r["top_gate"], z)
        lj = []
        for j in range(2):
            gs = lgate(r["sub_gates"][j], z)
            for ell in range(2):
                e = 2 * j + ell
                m = sum(c * v for c, v in zip(r["experts"][e], z))
                lj.append(
                    gt[j]
                    + gs[ell]
                    - 0.5 * (yi - m) ** 2 / r["sigma2"][e]
                    - 0.5 * math.log(2 * math.pi * r["sigma2"][e])
                )
        mx = max(lj)
        ex = [math.exp(v - mx) for v in lj]
        H.append([v / sum(ex) for v in ex])
    for e in range(4):
        for a in range(2):
            g = sum(H[i][e] * Z[i][a] * (y[i] - sum(c * v for c, v in zip(r["experts"][e], Z[i]))) for i in range(60))
            assert abs(g) < 1e-4


def test_classification_hme_is_monotone():
    X, y = data()
    yb = [1 if math.sin(3 * x[0]) + 0.3 * math.cos(9 * t) > 0 else 0 for x, t in zip(X, range(1, 61))]
    c = esl_hme(X, yb, K=2, task="classification", max_iter=60)
    p = c["loglik_path"]
    assert all(b >= a - 1e-8 for a, b in zip(p, p[1:])) and p[-1] > p[0]
