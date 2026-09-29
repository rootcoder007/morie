"""Tests for morie.fn.semres: every expected value is recomputed from the formula."""

from morie.fn.semres import semres

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def _moran(e, Wm):
    n = len(e)
    S0 = sum(map(sum, Wm))
    return n / S0 * sum(e[i] * Wm[i][j] * e[j] for i in range(n) for j in range(n)) / sum(v * v for v in e)


def _normal_moments(Wm):
    # Cliff-Ord moments under normality; equal to the exact residual moments when X is a constant
    n = len(Wm)
    S0 = sum(map(sum, Wm))
    S1 = 0.5 * sum((Wm[i][j] + Wm[j][i]) ** 2 for i in range(n) for j in range(n))
    S2 = sum((sum(Wm[i]) + sum(Wm[j][i] for j in range(n))) ** 2 for i in range(n))
    E = -1 / (n - 1)
    V = (n * n * S1 - n * S2 + 3 * S0 * S0) / (S0 * S0 * (n * n - 1)) - E * E
    return E, V


def test_filtered_residuals():
    lam = 0.3
    f = [E[i] - lam * sum(W[i][j] * E[j] for j in range(N)) for i in range(N)]
    assert abs(semres(E, W, lam).statistic - _moran(f, W)) < 1e-13


def test_filtered_constant_design_moments():
    # a row-standardised filter maps the constant to (1 - lam) * 1: same column space
    lam = 0.25
    m = sum(E) / N
    e0 = [v - m for v in E]
    # choose e so that the filtered residual is e0: e = (I - lam W)^{-1} e0 by fixed-point iteration
    e = list(e0)
    for _ in range(200):
        e = [e0[i] + lam * sum(W[i][j] * e[j] for j in range(N)) for i in range(N)]
    r = semres(e, W, lam, [[1.0] for _ in range(N)])
    Ex, V = _normal_moments(W)
    assert abs(r.statistic - _moran(e0, W)) < 1e-12
    assert abs(r.expected - Ex) < 1e-12
    assert abs(r.variance - V) < 1e-12
