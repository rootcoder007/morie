"""Wilcox (2017) eqs 4.9-4.11, 6.13, 7.36, 8.6, 8.12, 14.10 against WRS2 values and first principles."""

import math

from morie.fn.akpd import akpd
from morie.fn.ciovlap import ciovlap
from morie.fn.explpow import explpow
from morie.fn.r2ftest import r2ftest
from morie.fn.trimci import trimci
from morie.fn.trimse import trimse
from morie.fn.yuen import yuen

X = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.2, 2.2, 5.0, 40.0]
Y = [3.3, 4.1, 2.7, 6.8, 5.9, 4.4, 3.6, 7.2, 4.8, 5.3, 3.1, 6.6, 4.0, 9.5, 5.5]


def test_trimmed_mean_se_and_tukey_mclaughlin():
    r = trimse(X)
    # by hand: g = 2, Winsorized sample pulls 1.9, 2.1 up to 2.2 and 40, 6.1 down to 5.6
    w = sorted(min(max(v, 2.2), 5.6) for v in X)
    m = sum(w) / 13
    wv = sum((v - m) ** 2 for v in w) / 12
    assert abs(r["winsorized_variance"] - wv) < 1e-14 and abs(r["se"] - math.sqrt(wv) / (0.6 * math.sqrt(13))) < 1e-15
    assert abs(r["se"] - 0.607161042974655) < 1e-12  # WRS2::trimse
    assert abs(r["winsorized_variance"] - 1.72525641025641) < 1e-12  # WRS2::winvar
    t = trimci(X, null_value=3)
    assert t["df"] == 8 and abs(t["statistic"] - 1.42740822504128) < 1e-11
    assert abs(t["ci"][0] - 2.46655079083224) < 1e-9 and abs(t["ci"][1] - 5.26678254250109) < 1e-9


def test_yuen_matches_wrs2():
    r = yuen(X, Y)
    assert abs(abs(r["statistic"]) - 1.3652186507445) < 1e-11 and abs(r["df"] - 15.9951439167129) < 1e-10
    assert abs(r["diff"] + 1.04444444444444) < 1e-12
    assert abs(r["ci"][0] + 2.66629303568312) < 1e-9 and abs(r["ci"][1] - 0.577404146794235) < 1e-9
    assert abs(r["p_value"] - 0.191079509438389) < 1e-9


def test_akp_effect_size():
    r = akpd(X, null_value=3)
    assert abs(r["k"] - 0.641939815485996) < 1e-12  # the 0.642 of eq 8.6
    assert abs(r["d"] - 0.423564527123191) < 1e-12
    assert abs(akpd(X, tr=0)["k"] - 1) < 1e-15


def test_overlap_rule_f_from_r2_and_explanatory_power():
    r = ciovlap(X[:12], Y)
    assert r["welch_ratio"] >= r["ratio"]  # sum of standard errors >= root of summed squares
    xs = [1.0, 2.0, 3.0]
    ys = [10.0, 11.0, 12.0]
    e = 1 / math.sqrt(3)
    assert abs(ciovlap(xs, ys)["ratio"] - 9 / (2 * e)) < 1e-12 and ciovlap(xs, ys)["reject"]
    f = r2ftest(0.3, 40, 3)
    assert abs(f["statistic"] - 36 / 3 * 0.3 / 0.7) < 1e-12 and (f["df1"], f["df2"]) == (3, 36)
    # least-squares fitted values: eta^2 is R^2
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.2, 1.9, 3.4, 3.8, 5.3, 5.9]
    mx, my = sum(x) / 6, sum(y) / 6
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    fit = [my + b * (a - mx) for a in x]
    r2 = 1 - sum((c - f) ** 2 for c, f in zip(y, fit)) / sum((c - my) ** 2 for c in y)
    assert abs(explpow(y, fit)["eta2"] - r2) < 1e-12
