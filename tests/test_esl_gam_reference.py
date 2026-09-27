"""esl_gam against the penalised least-squares solution (R stacked system) and logit stationarity."""

import math

from morie.fn.eslgam import esl_gam
from morie.fn.nlsgn import _inverse


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    i = list(range(1, 41))
    X = [[((7 * t) % 41) / 41 * 3, ((11 * t) % 43) / 43 * 2] for t in i]
    return i, X


def penalty(x):
    """Reinsch penalty matrix K = Q R^-1 Q^T in data order."""
    o = sorted(range(len(x)), key=lambda k: x[k])
    u = [x[k] for k in o]
    n = len(u)
    h = [u[k + 1] - u[k] for k in range(n - 1)]
    Q = [[0.0] * (n - 2) for _ in range(n)]
    R = [[0.0] * (n - 2) for _ in range(n - 2)]
    for j in range(1, n - 1):
        Q[j - 1][j - 1], Q[j][j - 1], Q[j + 1][j - 1] = 1 / h[j - 1], -1 / h[j - 1] - 1 / h[j], 1 / h[j]
        R[j - 1][j - 1] = (h[j - 1] + h[j]) / 3
        if j < n - 2:
            R[j - 1][j] = R[j][j - 1] = h[j] / 6
    Ri = _inverse(R)
    Ks = [
        [sum(Q[a][c] * Ri[c][d] * Q[b][d] for c in range(n - 2) for d in range(n - 2)) for b in range(n)]
        for a in range(n)
    ]
    K = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            K[o[a]][o[b]] = Ks[a][b]
    return K


def test_identity_backfitting_equals_stacked_solution():
    i, X = data()
    y = [math.sin(2 * r[0]) + 0.5 * r[1] ** 2 + 0.2 * math.cos(9 * t) for r, t in zip(X, i)]
    r = esl_gam(X, y, lambdas=[0.05, 0.2])
    # qr.solve of the stacked system [I + l1 K1, I; I, I + l2 K2] f = (y - ybar, y - ybar) with centring rows
    ref0 = (0.78547241788421096, 0.82829556688955241, 0.048506283137461144)
    ref1 = (-0.51774963239376359, -0.095586835154616634, 0.55610053158069661)
    assert all(close(a, b, 1e-8) for a, b in zip(r["partial_fits"][0][:3], ref0))
    assert all(close(a, b, 1e-8) for a, b in zip(r["partial_fits"][1][:3], ref1))
    assert close(r["alpha"], 0.65983697203845071) and close(r["rss"], 1.0146439575827859, 1e-9)


def test_logit_local_scoring_stationarity():
    i, X = data()
    yb = [1 if math.sin(2 * r[0]) + r[1] - 1 + 0.8 * math.cos(9 * t) > 0 else 0 for r, t in zip(X, i)]
    q = esl_gam(X, yb, g="logit", lambdas=[0.5, 0.5])
    assert q["converged"]
    res = [a - b for a, b in zip(yb, q["fitted"])]
    assert abs(sum(res)) < 1e-8
    for j in range(2):
        K = penalty([r[j] for r in X])
        f = q["partial_fits"][j]
        assert max(abs(res[a] - 0.5 * sum(K[a][b] * f[b] for b in range(40))) for a in range(40)) < 1e-7


def test_linear_smoother_is_least_squares():
    i, X = data()
    y = [math.sin(2 * r[0]) + 0.5 * r[1] ** 2 for r in X]
    lin = esl_gam(X, y, smoother="linear")
    from morie.fn.linsys import _householder_ls

    beta, _ = _householder_ls([[1.0] + r for r in X], y)
    assert all(close(a, beta[0] + beta[1] * r[0] + beta[2] * r[1], 1e-9) for a, r in zip(lin["fitted"], X))


def test_logit_with_tied_inputs():
    # ties are pooled with their IRLS weights; stationarity then holds per distinct x:
    # sum over ties of (y - p) = lambda (K_u f_u) at that value
    i = list(range(1, 61))
    X = [[round(((7 * t) % 41) / 41 * 3, 1), round(((11 * t) % 43) / 43 * 2, 1)] for t in i]
    yb = [1 if math.sin(2 * r[0]) + r[1] - 1 + 0.8 * math.cos(9 * t) > 0 else 0 for r, t in zip(X, i)]
    q = esl_gam(X, yb, g="logit", lambdas=[0.3, 0.3])
    assert q["converged"]
    res = [a - b for a, b in zip(yb, q["fitted"])]
    for j in range(2):
        xs = [r[j] for r in X]
        u = sorted(set(xs))
        K = penalty(u)
        fu = [q["partial_fits"][j][xs.index(v)] for v in u]
        agg = [sum(res[k] for k in range(60) if xs[k] == v) for v in u]
        assert max(abs(agg[a] - 0.3 * sum(K[a][b] * fu[b] for b in range(len(u)))) for a in range(len(u))) < 1e-7
