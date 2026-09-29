"""Tests for obssens: Rosenbaum sensitivity analysis for matched observational studies."""

import math

from morie.fn.obssens import (
    amplify_gamma,
    crosscut_design_sensitivity,
    crosscut_test,
    gamma_from_lambda_delta,
    u_statistic_sensitivity,
)


def test_amplification_round_trip():
    r = gamma_from_lambda_delta([1.5, 2.0, 7.0], [2.5, 3.0, 1.2])
    assert r.gamma == [(a * b + 1) / (a + b) for a, b in ((1.5, 2.5), (2.0, 3.0), (7.0, 1.2))]
    assert r.abz_bounds[1] == [0.25, 0.75]
    for lam in (1.8, 2.5, 10.0):
        dl = amplify_gamma(1.7, [lam])[0]
        assert abs((lam * dl + 1) / (lam + dl) - 1.7) <= 1e-12
    assert math.isnan(amplify_gamma(2.0, [1.5])[0])


def _pn(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def test_u_statistic_wilcoxon_case_is_signed_rank_bound():
    d = [math.sin(i * 1.7) * 2 + 0.6 + 0.1 * (i % 4) for i in range(30)]
    g = 1.3
    r = u_statistic_sensitivity(d, g)  # (2, 2, 2): scores (q - 1) / C(I, 2)
    ad = [abs(v) for v in d]
    q = [sorted(ad).index(a) + 1 for a in ad]
    sc = [(k - 1) / (30 * 29 / 2) for k in q]
    pr = g / (1 + g)
    t = sum(s for s, v in zip(sc, d) if v > 0)
    z = (t - pr * sum(sc)) / math.sqrt(sum(s * s for s in sc) * pr * (1 - pr))
    assert abs(r.p_value - (1 - _pn(z))) <= 1e-14
    assert abs(r.statistic - t) <= 1e-14


def test_crosscut_exact_hypergeometric_at_gamma_one():
    x = [(i * 37) % 101 + 0.5 * math.sin(i) for i in range(120)]
    y = [0.02 * x[i] + math.cos(i * 2.1) for i in range(120)]
    r = crosscut_test(x, y, 0.25)
    (a, b), (c, e) = r.table
    m1, m2, n = a + b, c + e, a + c

    def ch(nn, kk):
        return math.comb(nn, kk) if 0 <= kk <= nn else 0

    tot = sum(ch(m1, k) * ch(m2, n - k) for k in range(0, n + 1))
    want = sum(ch(m1, k) * ch(m2, n - k) for k in range(a, n + 1)) / tot
    assert abs(r.p_value - want) <= 1e-12
    assert crosscut_test(x, y, 0.25, 2.0).p_value > r.p_value


def test_design_sensitivity_reproduces_dos_table_19_2():
    table = {0.5: [1.3, 2.2, 4.0], 0.25: [1.9, 7.7, 44.5], 0.2: [2.2, 12.0, 106.7], 0.125: [3.0, 32.1, 740.4]}
    for eta, row in table.items():
        got = crosscut_design_sensitivity([0.1, 0.3, 0.5], [eta] * 3)
        assert [round(v, 1) for v in got] == row
    # at the median the corner odds ratio has Sheppard's closed form
    for rho in (0.2, 0.7):
        p = 0.25 + math.asin(rho) / (2 * math.pi)
        assert abs(crosscut_design_sensitivity([rho], [0.5])[0] - (p / (0.5 - p)) ** 2) <= 1e-12
