"""Tests for tmlfed.tmle_federated (influence-curve-weighted pooling)."""

import math
import statistics

import pytest

from morie.fn.tmlfed import tmle_federated, tmlefederated


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _site(y, D, X, gb=0.025):
    """Logistic propensity by Newton to convergence, OLS outcome model
    Y ~ D + 1 + X by normal equations, closed-form linear fluctuation on
    H = D/g - (1-D)/(1-g) (squared-error loss), IC at the targeted fit."""
    n = len(y)
    W = [[1.0] + x for x in X]
    p = len(W[0])
    b = [0.0] * p
    for _ in range(60):
        mu = [_expit(sum(a * c for a, c in zip(w, b))) for w in W]
        I = [[sum(W[i][r] * W[i][s] * mu[i] * (1 - mu[i]) for i in range(n)) for s in range(p)] for r in range(p)]
        sc = [sum(W[i][r] * (D[i] - mu[i]) for i in range(n)) for r in range(p)]
        b = [x + d for x, d in zip(b, _solve(I, sc))]
    g = [min(max(_expit(sum(a * c for a, c in zip(w, b))), gb), 1 - gb) for w in W]
    Z = [[d] + w for d, w in zip(D, W)]
    q = _solve([[sum(z[r] * z[s] for z in Z) for s in range(p + 1)] for r in range(p + 1)],
               [sum(z[r] * t for z, t in zip(Z, y)) for r in range(p + 1)])
    Q1 = [q[0] + sum(a * c for a, c in zip(w, q[1:])) for w in W]
    Q0 = [sum(a * c for a, c in zip(w, q[1:])) for w in W]
    H = [d / e - (1 - d) / (1 - e) for d, e in zip(D, g)]
    QA = [u if d else v for d, u, v in zip(D, Q1, Q0)]
    eps = sum(h * (t - qa) for h, t, qa in zip(H, y, QA)) / sum(h * h for h in H)
    Q1s = [u + eps / e for u, e in zip(Q1, g)]
    Q0s = [v - eps / (1 - e) for v, e in zip(Q0, g)]
    psi = statistics.fmean(u - v for u, v in zip(Q1s, Q0s))
    ic = [h * (t - qa - eps * h) + u - v - psi for h, t, qa, u, v in zip(H, y, QA, Q1s, Q0s)]
    return psi, statistics.variance(ic)


def _data(n=90):
    X = [[math.sin(0.9 * k), math.cos(1.3 * k)] for k in range(n)]
    D = [1.0 if ((43 * k + 1) % 97 + 0.5) / 97.0 < _expit(0.5 * x[0]) else 0.0 for k, x in enumerate(X)]
    y = [1.0 + 0.7 * d + 0.4 * x[0] - 0.3 * x[1] + 0.3 * math.sin(7.1 * k) for k, (d, x) in enumerate(zip(D, X))]
    site = [k % 3 for k in range(n)]
    return y, D, X, site


def test_tmlfed_basic():
    """Site estimates match an independent per-site TMLE; the pooled
    estimate is sum w_s psi_s / sum w_s with w_s = n_s / var(IC_s) and
    se = 1/sqrt(sum w_s).  Tolerance 1e-8: the module's IRLS carries a
    1e-8 ridge, which moves the propensity coefficients by at most
    1e-8/lambda_min(X'WX), and lambda_min here is far above 1."""
    y, D, X, site = _data()
    fits, ns = [], []
    for s in (0, 1, 2):
        idx = [i for i in range(len(y)) if site[i] == s]
        fits.append(_site([y[i] for i in idx], [D[i] for i in idx], [X[i] for i in idx]))
        ns.append(len(idx))
    w = [n / v for (_, v), n in zip(fits, ns)]
    pooled = sum(wi * f[0] for wi, f in zip(w, fits)) / sum(w)
    r = tmle_federated(y, D, X, site)
    assert r["site_psi"] == pytest.approx([f[0] for f in fits], abs=1e-8)
    assert r["estimate"] == pytest.approx(pooled, abs=1e-8)
    assert r["se"] == pytest.approx(1.0 / math.sqrt(sum(w)), rel=1e-8)
    assert r["site_n"] == ns and r["n_sites"] == 3 and r["n"] == len(y)


def test_tmlfed_edge():
    """One site is the site's own TMLE with its IC standard error."""
    y, D, X, _ = _data()
    psi, v = _site(y, D, X)
    r = tmlefederated(y, D, X, [7] * len(y))
    assert r["estimate"] == pytest.approx(psi, abs=1e-8)
    assert r["se"] == pytest.approx(math.sqrt(v / len(y)), rel=1e-8)
