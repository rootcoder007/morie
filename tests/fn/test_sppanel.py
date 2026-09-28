"""Tests for sppanel: spatial panel data models."""

import math

from morie.fn.sppanel import sp_panel_dynamic, sp_panel_fe, sp_panel_re

N, T = 8, 5
_A = [[1.0 if (i != j and (abs(i - j) == 1 or (i * 3 + j * 5) % 7 == 0)) else 0.0 for j in range(N)] for i in range(N)]
_A = [[max(_A[i][j], _A[j][i]) for j in range(N)] for i in range(N)]
W = [[a / sum(r) for a in r] for r in _A]
X = [[[math.sin(i + 2 * t), math.cos(i * t * 0.3) + 0.1 * i] for i in range(N)] for t in range(T)]
Y = [
    [0.7 * X[t][i][0] - 0.4 * X[t][i][1] + 0.3 * math.sin(i * 1.7) + 0.2 * math.cos(t * i) for i in range(N)]
    for t in range(T)
]


def _det(M):
    A = [list(r) for r in M]
    n, d = len(A), 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return d


def _ols_resid(Xc, y):
    k = len(Xc)
    G = [[sum(a * b for a, b in zip(Xc[i], Xc[j])) for j in range(k)] for i in range(k)]
    h = [sum(a * b for a, b in zip(Xc[i], y)) for i in range(k)]
    # 2 x 2 or 4 x 4 solve by Gauss-Jordan
    M = [G[i] + [h[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    b = [M[i][k] / M[i][i] for i in range(k)]
    return b, [v - sum(Xc[j][r] * b[j] for j in range(k)) for r, v in enumerate(y)]


def _within(v):
    um = [sum(v[t * N + i] for t in range(T)) / T for i in range(N)]
    return [v[t * N + i] - um[i] for t in range(T) for i in range(N)]


def _lag_ll(rho):
    y = [Y[t][i] for t in range(T) for i in range(N)]
    wy = [sum(W[i][j] * Y[t][j] for j in range(N)) for t in range(T) for i in range(N)]
    Xc = [_within([X[t][i][k] for t in range(T) for i in range(N)]) for k in range(2)]
    _, e = _ols_resid(Xc, [a - rho * b for a, b in zip(_within(y), _within(wy))])
    ld = math.log(abs(_det([[(i == j) - rho * W[i][j] for j in range(N)] for i in range(N)])))
    return -0.5 * N * T * math.log(sum(v * v for v in e)) + T * ld, Xc, e


def test_fe_lag_maximises_concentrated_likelihood():
    r = sp_panel_fe(Y, X, W)
    l0 = _lag_ll(r.rho)[0]
    assert l0 >= _lag_ll(r.rho + 1e-4)[0] and l0 >= _lag_ll(r.rho - 1e-4)[0]
    _, Xc, e = _lag_ll(r.rho)
    assert max(abs(a - b) for a, b in zip(e, r.residuals)) <= 1e-10
    s2 = sum(v * v for v in e) / (N * T)
    ld = math.log(abs(_det([[(i == j) - r.rho * W[i][j] for j in range(N)] for i in range(N)])))
    assert abs(r.loglik - (-0.5 * N * T * (math.log(2 * math.pi * s2) + 1) + T * ld)) <= 1e-9


def test_fe_error_and_durbin_are_local_maxima():
    for model in ("error", "durbin"):
        r = sp_panel_fe(Y, X, W, model=model, effects="twoways")
        p = r.get("rho", r.get("lambda_"))
        lo = sp_panel_fe(Y, X, W, model=model, effects="twoways", bounds=(p - 1e-3, p - 1e-3 + 1e-12))
        assert r.loglik >= lo.loglik


def _re_ll(phi, rho):
    th = 1 - 1 / math.sqrt(1 + T * phi)
    y = [Y[t][i] - rho * sum(W[i][j] * Y[t][j] for j in range(N)) for t in range(T) for i in range(N)]

    def qd(v):
        um = [sum(v[t * N + i] for t in range(T)) / T for i in range(N)]
        return [v[t * N + i] - th * um[i] for t in range(T) for i in range(N)]

    Xc = [qd([1.0] * (N * T))] + [qd([X[t][i][k] for t in range(T) for i in range(N)]) for k in range(2)]
    _, e = _ols_resid(Xc, qd(y))
    s2 = sum(v * v for v in e) / (N * T)
    ld = math.log(abs(_det([[(i == j) - rho * W[i][j] for j in range(N)] for i in range(N)])))
    return -0.5 * N * T * (math.log(2 * math.pi * s2) + 1) - 0.5 * N * math.log(1 + T * phi) + T * ld


def test_re_lag_is_stationary():
    r = sp_panel_re(Y, X, W)
    assert abs(r.loglik - _re_ll(r.phi, r.rho)) <= 1e-9
    for dp, dr in ((1e-5, 0), (0, 1e-5)):
        if r.phi > 1e-4 or dr:
            g = (_re_ll(r.phi + dp, r.rho + dr) - _re_ll(r.phi - dp, r.rho - dr)) / 2e-5
            assert abs(g) <= 1e-4


def test_dynamic_is_lag_model_with_lagged_regressors():
    d = sp_panel_dynamic(Y, X, W)
    Xn = [
        [[Y[t - 1][i], sum(W[i][j] * Y[t - 1][j] for j in range(N))] + X[t][i] for i in range(N)] for t in range(1, T)
    ]
    r = sp_panel_fe(Y[1:], Xn, W)
    assert abs(r.rho - d.rho) <= 1e-12
    assert max(abs(a - b) for a, b in zip(r.beta, d.beta)) <= 1e-12
