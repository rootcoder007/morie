import math
import statistics

import pytest

from morie.fn.envstats import budyko_olr, empirical_breakdown_point, fleiss_kappa, prewhitened_mann_kendall


def test_budyko():
    r = budyko_olr(12.5, albedo=0.31)
    assert r.olr == pytest.approx(203.3 + 2.09 * 12.5, abs=1e-12)
    assert r.equilibrium_temperature == pytest.approx((0.69 * 1361 / 4 - 203.3) / 2.09, abs=1e-12)
    assert r.sensitivity == pytest.approx(1 / 2.09, abs=1e-12)


def _mk_brute(y):
    n = len(y)
    S = sum((y[j] > y[i]) - (y[j] < y[i]) for i in range(n) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    return S, var


def test_pw_recomputes():
    x = [1.0, 2.1, 2.9, 4.2, 5.1, 5.8, 7.2, 8.1, 7.9, 9.4]
    r = prewhitened_mann_kendall(x, "pw")
    m = statistics.fmean(x)
    d = [v - m for v in x]
    r1 = sum(d[t] * d[t + 1] for t in range(9)) / sum(v * v for v in d)
    y = [x[t + 1] - r1 * x[t] for t in range(9)]
    S, var = _mk_brute(y)
    assert r.r1 == pytest.approx(r1, abs=1e-12)
    assert r.S == S and r.var_S == pytest.approx(var, abs=1e-9)
    assert pytest.approx((S - math.copysign(1, S)) / math.sqrt(var), abs=1e-12) == r.Z
    slopes = [(y[j] - y[i]) / (j - i) for i in range(9) for j in range(i + 1, 9)]
    assert r.sen_slope == pytest.approx(statistics.median(slopes), abs=1e-12)


def test_tfpw_trend_detected():
    x = [0.3 * t + ((-1) ** t) * 0.2 for t in range(20)]
    r = prewhitened_mann_kendall(x, "tfpw")
    assert r.S > 0 and r.p_value < 1e-6
    with pytest.raises(ValueError):
        prewhitened_mann_kendall(x, "bad")


def test_fleiss_formula():
    C = [[3, 1, 0], [0, 2, 2], [4, 0, 0], [1, 1, 2], [0, 0, 4], [2, 2, 0]]
    r = fleiss_kappa(C)
    N, m = 6, 4
    P = [(sum(v * v for v in row) - m) / (m * (m - 1)) for row in C]
    pj = [sum(row[j] for row in C) / (N * m) for j in range(3)]
    pe = sum(p * p for p in pj)
    assert r.kappa == pytest.approx((sum(P) / N - pe) / (1 - pe), abs=1e-12)
    assert fleiss_kappa([[2, 0], [0, 2]]).kappa == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [1, 2]])


def test_breakdown():
    x = [float(v) for v in range(1, 12)]
    assert empirical_breakdown_point(lambda v: sum(v) / len(v), x).m == 1
    assert empirical_breakdown_point(statistics.median, x).m == 6
