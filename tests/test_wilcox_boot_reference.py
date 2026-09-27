"""Wilcox (2017) eqs 7.15, 7.39, 11.6, 13.2, 15.15 and Sec 12.1.13: recomputation and R parity."""

import math

from morie.fn._rng import random_uniform
from morie.fn.comvar2 import comvar2
from morie.fn.friedf import friedf
from morie.fn.linconbt import linconbt
from morie.fn.logrsm import logrsm
from morie.fn.outmah import outmah
from morie.fn.pbci import pbci
from morie.fn.trimse import _tmean
from morie.fn.yuen import _yuen_d

X = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.2, 2.2, 5.0, 40.0]
Y = [3.3, 4.1, 2.7, 6.8, 5.9, 4.4, 3.6, 7.2, 4.8, 5.3, 3.1, 6.6, 4.0, 9.5, 5.5]
Z = [5.1, 6.3, 4.8, 7.7, 6.9, 5.5, 6.1, 8.4, 5.9, 7.0, 4.6]


def test_percentile_interval_indices():
    v = [((7 * i) % 101) - 50.0 for i in range(101)]  # a permutation of -50..50
    r = pbci(v, alpha=0.1)
    lo = round(0.1 * 101 / 2)  # 5
    assert (r["l"], r["u"]) == (lo, 101 - lo) and r["ci"] == (-50.0 + lo, -50.0 + 101 - lo - 1)
    assert abs(r["p_value"] - 2 * min(50.5 / 101, 1 - 50.5 / 101)) < 1e-15


def test_variance_difference_bootstrap():
    r = comvar2(X, Y, seed=3)
    nm = 13
    ux = random_uniform(599 * nm, seed=3, stream=0)
    uy = random_uniform(599 * nm, seed=3, stream=1)

    def var(v):
        m = sum(v) / len(v)
        return sum((a - m) ** 2 for a in v) / (len(v) - 1)

    d = sorted(
        var([X[math.floor(ux[b * nm + i] * 13)] for i in range(nm)])
        - var([Y[math.floor(uy[b * nm + i] * 15)] for i in range(nm)])
        for b in range(599)
    )
    assert (r["l"], r["u"]) == (7, 593) and r["ci"] == (d[6], d[592])
    assert abs(r["ci"][0] + 3.7861538461538458) < 1e-12 and abs(r["ci"][1] - 261.30641025641023) < 1e-9  # R arm


def test_bootstrap_t_tmax():
    r = linconbt([X, Y, Z], seed=5)
    assert r["crit"] == r["tmax"][round(0.95 * 599) - 1]
    est = [_tmean(g, 0.2) for g in (X, Y, Z)]
    d = [_yuen_d(g, 0.2)[0] for g in (X, Y, Z)]
    for c in r["comparisons"]:
        se = math.sqrt(d[c["j"]] + d[c["k"]])
        assert (
            abs(c["diff"] - (est[c["j"]] - est[c["k"]])) < 1e-15
            and abs(c["lower"] - (c["diff"] - r["crit"] * se)) < 1e-14
        )
    assert abs(r["crit"] - 2.7815908479889098) < 1e-12  # R arm, same Philox streams
    assert abs(r["comparisons"][1]["lower"] + 4.0979961070047697) < 1e-12


def test_friedman_f_form():
    T = [[9, 7, 12], [1, 10, 4], [8, 2, 1], [5, 6, 9], [3, 3, 7], [6, 8, 8], [2, 9, 5]]
    r = friedf(T)
    n, J = 7, 3
    # with ties the F form uses A = sum of squared midranks; compare with the Friedman chi-square it implies
    Rj = [0.0] * J
    for row in T:
        s = sorted(row)
        for j, v in enumerate(row):
            Rj[j] += (s.index(v) + len(s) - s[::-1].index(v)) / 2 + 0.5
    assert abs(r["B"] - sum(v * v for v in Rj) / n) < 1e-12 and r["C"] == n * J * (J + 1) ** 2 / 4
    assert abs(r["statistic"] - 1.6363636363636365) < 1e-12  # Iman-Davenport F from R's friedman.test chi-square
    assert abs(r["p_value"] - 0.2352815631667069) < 1e-12


def test_outlier_rule_and_binary_smoother():
    P = [[1, 2], [2, 3], [3, 5], [4, 4], [5, 6], [6, 7], [7, 8], [20, -5], [3, 3], [4, 6]]
    r = outmah(P, center=[0, 0], scatter=[[4, 0], [0, 1]])
    assert (
        abs(r["distance"][1] - math.sqrt(4 / 4 + 9)) < 1e-15 and abs(r["crit"] - math.sqrt(7.377758908227871)) < 1e-12
    )
    assert outmah(P)["outliers"] == [7]  # MVE center and scatter, as the R arm
    xs = [float(i) for i in range(1, 31)]
    ys = [1.0 if (i * 7) % 10 < i / 3 else 0.0 for i in range(1, 31)]
    m = 15.5
    madn = 7.5 / 0.6745
    z0 = (15 - m) / madn
    w = [math.exp(-(((a - m) / madn - z0) ** 2)) if abs((a - m) / madn - z0) < 1.2 else 0.0 for a in xs]
    ph = logrsm(xs, ys, pts=[5.0, 15.0, 25.0])["phat"]
    assert abs(ph[1] - sum(a * b for a, b in zip(w, ys)) / sum(w)) < 1e-15
    assert abs(ph[0] - 0.13471408906394922) < 1e-12 and abs(ph[2] - 0.63222862469641816) < 1e-12
