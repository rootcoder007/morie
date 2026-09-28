"""paneldiag: identities recomputed from the defining formulas (plm/fixest agreement is in the R cross test)."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.paneldiag import (
    baltagi_li_test,
    conley_vcov,
    cross_section_dependence,
    panel_residuals,
    panel_serial_test,
    panel_variance_components,
    panel_within,
    unobserved_effects_test,
)

N, T = 6, 5
_e = random_normal(3 * N * T, seed=5)
UNIT = [i // T for i in range(N * T)]
TIME = [i % T for i in range(N * T)]
X = [[_e[i] + 0.2 * TIME[i], _e[N * T + i] * 0.5 + UNIT[i]] for i in range(N * T)]
Y = [1.0 + 0.4 * X[i][0] - 0.3 * X[i][1] + math.cos(UNIT[i]) + _e[2 * N * T + i] for i in range(N * T)]


def _unit_rows(u):
    return [i for i in range(N * T) if UNIT[i] == u]


def _pearson(a, b):
    n = len(a)
    sa, sb = sum(a), sum(b)
    num = n * sum(x * y for x, y in zip(a, b)) - sa * sb
    return num / math.sqrt((n * sum(x * x for x in a) - sa * sa) * (n * sum(y * y for y in b) - sb * sb))


def test_within_and_variance_components():
    w = panel_within(X, UNIT)
    for u in range(N):
        rows = _unit_rows(u)
        for j in range(2):
            m = sum(X[i][j] for i in rows) / T
            for i in rows:
                assert w[i][j] == pytest.approx(X[i][j] - m, abs=1e-12)
    e = panel_residuals(Y, X, UNIT, "pooling").residuals
    vc = panel_variance_components(e, UNIT)
    means = [sum(e[i] for i in _unit_rows(u)) / T for u in range(N)]
    q = sum((e[i] - means[UNIT[i]]) ** 2 for i in range(N * T))
    assert vc.sigma2_idios == pytest.approx(q / (N * (T - 1)), rel=1e-12)
    assert vc.sigma2_1 == pytest.approx(T * sum(m * m for m in means) / N, rel=1e-12)
    assert vc.theta == pytest.approx(1 - math.sqrt(vc.sigma2_idios / vc.sigma2_1), rel=1e-12)
    with pytest.raises(ValueError):
        panel_variance_components(e[:-1], UNIT[:-1])


def test_residual_models_are_orthogonal():
    e = panel_residuals(Y, X, UNIT, "pooling").residuals
    assert abs(sum(e)) < 1e-10 and abs(sum(e[i] * X[i][0] for i in range(N * T))) < 1e-9
    e = panel_residuals(Y, X, UNIT, "within").residuals
    assert abs(sum(e[i] * panel_within(X, UNIT)[i][1] for i in range(N * T))) < 1e-9
    h = panel_residuals(Y, X, UNIT, "heterogeneous").residuals
    for u in range(N):
        rows = _unit_rows(u)
        assert abs(sum(h[i] for i in rows)) < 1e-10
        assert abs(sum(h[i] * X[i][1] for i in rows)) < 1e-9


def test_cross_section_dependence_by_hand():
    e = panel_residuals(Y, X, UNIT, "heterogeneous").residuals
    rho = [
        _pearson([e[i] for i in _unit_rows(a)], [e[i] for i in _unit_rows(b)])
        for b in range(N)
        for a in range(b + 1, N)
    ]
    m = len(rho)
    assert cross_section_dependence(e, UNIT, TIME).statistic == pytest.approx(math.sqrt(T / m) * sum(rho), rel=1e-10)
    assert cross_section_dependence(e, UNIT, TIME, "lm").statistic == pytest.approx(
        T * sum(r * r for r in rho), rel=1e-10
    )
    assert cross_section_dependence(e, UNIT, TIME, "absrho").statistic == pytest.approx(
        sum(abs(r) for r in rho) / m, rel=1e-10
    )
    sc = cross_section_dependence(e, UNIT, TIME, "sclm").statistic
    assert sc == pytest.approx(sum(T * r * r - 1 for r in rho) / math.sqrt(2 * m), rel=1e-10)
    wmat = [[1.0 if abs(a - b) == 1 else 0.0 for b in range(N)] for a in range(N)]
    loc = cross_section_dependence(e, UNIT, TIME, "cd", w=wmat)
    near = [_pearson([e[i] for i in _unit_rows(a + 1)], [e[i] for i in _unit_rows(a)]) for a in range(N - 1)]
    assert loc.n_pairs == N - 1
    assert loc.statistic == pytest.approx(math.sqrt(T / (N - 1)) * sum(near), rel=1e-10)


def test_unobserved_effects_identity():
    e = panel_residuals(Y, X, UNIT, "pooling").residuals
    S = []
    for u in range(N):
        v = [e[i] for i in _unit_rows(u)]
        S.append((sum(v) ** 2 - sum(x * x for x in v)) / 2)
    r = unobserved_effects_test(e, UNIT)
    assert r.statistic == pytest.approx(sum(S) / math.sqrt(sum(s * s for s in S)), rel=1e-10)


def _inv3(a):
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    return (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det


def test_baltagi_li_ml_and_information():
    two = baltagi_li_test(Y, X, UNIT)
    one = baltagi_li_test(Y, X, UNIT, "onesided")
    assert one.statistic**2 == pytest.approx(two.statistic, rel=1e-10)
    # the ML variance ratio solves Breusch's first-order condition at the GLS coefficients
    b = two.coefficients
    d = [Y[i] - b[0] - b[1] * X[i][0] - b[2] * X[i][1] for i in range(N * T)]
    md = [sum(d[i] for i in _unit_rows(u)) / T for u in range(N)]
    q = sum((d[i] - md[UNIT[i]]) ** 2 for i in range(N * T))
    assert two.phi2 == pytest.approx(min(1.0, q / ((T - 1) * T * sum(v * v for v in md))), rel=1e-10)
    s2e, s21 = two.sigma2_e, two.sigma2_1
    a = (s2e - s21) / (T * s21)
    J = [
        [
            N * (2 * a * a * (T - 1) ** 2 + 2 * a * (2 * T - 3) + T - 1),
            N * (T - 1) * s2e / s21**2,
            N * (T - 1) / T * s2e * (1 / s21**2 - 1 / s2e**2),
        ],
        [N * (T - 1) * s2e / s21**2, N * T * T / (2 * s21**2), N * T / (2 * s21**2)],
        [
            N * (T - 1) / T * s2e * (1 / s21**2 - 1 / s2e**2),
            N * T / (2 * s21**2),
            N / 2 * (1 / s21**2 + (T - 1) / s2e**2),
        ],
    ]
    assert pytest.approx(_inv3(J), rel=1e-10) == two.J11
    assert two.statistic == pytest.approx(two.D**2 * two.J11, rel=1e-12)


def test_panel_serial_chisq_equals_f_form():
    for order in (1, 2):
        c = panel_serial_test(Y, X, UNIT, order)
        f = panel_serial_test(Y, X, UNIT, order, "F")
        n, k = N * T, 2
        ratio = 1 / (1 + f.statistic * order / (n - k - order))
        assert c.statistic == pytest.approx(n * (1 - ratio), rel=1e-9)
        assert f.df == (order, n - k - order)


def test_conley_limits():
    lat = [44.0 + 0.1 * i for i in range(N * T)]
    lon = [-79.0 + 0.05 * (i % 7) for i in range(N * T)]
    e = panel_residuals(Y, X, UNIT, "pooling").residuals
    z = conley_vcov(X, e, lat, lon, 1e-9)
    Xi = [[1.0] + r for r in X]
    # cutoff zero: the White HC0 sandwich with the n/(n-k) factor
    from morie.fn._qpcore import inverse

    B = inverse([[sum(r[a] * r[b] for r in Xi) for b in range(3)] for a in range(3)])
    M = [[sum(Xi[i][a] * Xi[i][b] * e[i] ** 2 for i in range(N * T)) for b in range(3)] for a in range(3)]
    V = [[sum(B[a][c] * M[c][d] * B[d][b] for c in range(3) for d in range(3)) for b in range(3)] for a in range(3)]
    for a in range(3):
        assert z.vcov[a][a] == pytest.approx(V[a][a] * (N * T) / (N * T - 3), rel=1e-10)
    # cutoff beyond every distance: OLS scores sum to zero, so the covariance vanishes
    big = conley_vcov(X, e, lat, lon, 1e5)
    assert max(abs(v) for r in big.vcov for v in r) < 1e-12 * max(abs(v) for r in z.vcov for v in r) + 1e-12
    # Bartlett weights by hand on three points
    la, lo = [0.0, 0.0, 0.0], [0.0, 0.3, 0.9]
    r = conley_vcov(
        [[0.0], [1.0], [3.0]], [1.0, -2.0, 1.0], la, lo, 80.0, kernel="bartlett", adjust=False, intercept=False
    )
    d = [[6371 * math.radians(abs(lo[i] - lo[j])) for j in range(3)] for i in range(3)]
    s = [0.0, -2.0, 3.0]
    meat = sum(s[i] * s[j] * max(1 - d[i][j] / 80.0, 0.0) for i in range(3) for j in range(3))
    assert r.vcov[0][0] == pytest.approx(meat / 10.0**2, rel=1e-10)
