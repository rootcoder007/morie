"""Hedderich methods (batch 4) against R: AcceptanceSampling::find.plan, qqnorm, drop1, loglin, qr.solve."""

import math

from morie.fn.accpln import acceptance_sampling_plan
from morie.fn.accqlv import acceptance_quality_levels
from morie.fn.bcppcc import box_cox_ppcc
from morie.fn.bkelim import backward_elimination
from morie.fn.chisqmc import chisq_monte_carlo
from morie.fn.linsys import solve_linear_system
from morie.fn.ll3way import loglinear_three_way
from morie.fn.schomg import score_homogeneity_test


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_acceptance_quality_levels():
    r = acceptance_quality_levels(46, 1, lot_size=1000)
    # book Fig 7.3/7.4: AQL 0.0077, RQL 0.0819, AOQL 0.0174
    assert close(r["aql"], 0.0077802465479319639) and close(r["rql"], 0.0819470508731946490)
    assert close(r["aoql"], 0.017305563505389011) and close(r["p_aoql"], 0.034506093285850907, 1e-6)


def test_acceptance_sampling_plan():  # AcceptanceSampling::find.plan
    r = acceptance_sampling_plan(0.0077, 0.0819)
    assert (r["n"], r["c"]) == (64, 2)
    r = acceptance_sampling_plan(0.01, 0.06)
    assert (r["n"], r["c"]) == (110, 3)
    r = acceptance_sampling_plan(0.02, 0.10, alpha=0.01, beta_risk=0.05)
    assert (r["n"], r["c"]) == (116, 6)


def test_box_cox_ppcc():
    r = box_cox_ppcc([20, 22, 24, 21, 19, 30, 40, 23, 24, 25, 19])
    assert r["lambda"] == -5 + 20 * 0.1  # book p. 477: -3
    for k, v in zip(
        (0, 20, 50, 70), (0.97075861011113906, 0.98495222774565838, 0.92657242289873065, 0.82882425530178350)
    ):
        assert close(r["cor"][k], v)
    r = box_cox_ppcc([3.1, 0.4, 7.9, 1.2, 2.2, 5.5, 0.9], [-1, 0, 0.5, 1])
    for k, v in enumerate((0.89213850349800383, 0.99366574276927155, 0.98340414167770396, 0.94462963339686346)):
        assert close(r["cor"][k], v)
    assert r["lambda"] == 0


def test_score_homogeneity():
    r = score_homogeneity_test([[14, 22, 32], [18, 16, 8], [8, 2, 0]], [1, 0, -1])
    # book p. 742: 20.164; (n - 1) R^2 of lm(score ~ group)
    assert close(r["statistic"], 20.16410940627085) and r["df"] == 2


def test_chisq_monte_carlo():
    r = chisq_monte_carlo([[3, 1], [1, 3]], B=20000, seed=11)
    # exact conditional P(X^2 >= 2) with margins 4/4: a in {0, 1, 3, 4}, (1 + 16 + 16 + 1)/70
    assert close(r["statistic"], 2.0)
    assert abs(r["p_value"] - 34 / 70) < 4 * math.sqrt(34 / 70 * 36 / 70 / 20000)
    assert close(r["p_value"] * 20001, round(r["p_value"] * 20001))
    a = chisq_monte_carlo([[14, 22, 32], [18, 16, 8], [8, 2, 0]], B=1000, seed=3)
    assert a == chisq_monte_carlo([[14, 22, 32], [18, 16, 8], [8, 2, 0]], B=1000, seed=3)
    assert close(a["statistic"], 21.576470588235292) and a["p_value"] < 0.01  # book: 0.000999


def test_backward_elimination():
    i = range(1, 31)
    X = [[math.sin(t), math.cos(2 * t), (t % 7) / 7, math.log(t), ((t * 13) % 11) / 11] for t in i]
    y = [1 + 2 * r[0] + 0.05 * r[2] - 0.8 * r[3] + 0.3 * math.cos(5 * t) for r, t in zip(X, i)]
    r = backward_elimination(X, y)
    assert [n for n, _ in r["removed"]] == ["x2", "x3"] and r["selected"] == ["x1", "x4", "x5"]
    assert close(r["removed"][1][1], 4.1419658248545488e-02)  # drop1(lm(y ~ x1 + x3 + x4 + x5), test = "F")
    assert close(r["last_f"]["x5"], 5.7924148194401033) and close(r["last_f"]["x1"], 1260.8843911205676704)
    for a, b in zip(
        r["coefficients"], (0.86264391837289833, 1.90435155816806656, -0.79396954246054274, 0.32609513999726958)
    ):
        assert close(a, b, 1e-11)
    assert close(r["rss"], 1.04313738206858853, 1e-11)


def test_loglinear_three_way():
    v = [12, 7, 9, 15, 4, 11, 20, 6, 8, 13, 5, 10]
    t = [[[v[i + 2 * j + 6 * k] for k in range(2)] for j in range(3)] for i in range(2)]
    cases = [
        (
            [(0,), (1,), (2,)],
            18.125174054366685,
            18.399927740747657,
            7,
            (10.5125, 10.5125, 7.0083333333333346, 11.2375, 11.2375, 7.4916666666666663),
        ),
        (
            [(0, 1), (2,)],
            2.3258276966975711,
            2.3065634228364051,
            5,
            (15.466666666666667, 8.2166666666666668, 4.35, 16.533333333333335, 8.7833333333333332, 4.65),
        ),
        (
            [(0, 1), (0, 2)],
            1.0939678707576046,
            1.0965435045759586,
            4,
            (
                13.793103448275861,
                7.3275862068965516,
                3.8793103448275859,
                18.206896551724139,
                9.6724137931034484,
                5.1206896551724137,
            ),
        ),
        (
            [(0, 1), (0, 2), (1, 2)],
            0.49883993665169152,
            0.49891589940206360,
            2,
            (
                12.782141396395414,
                8.2217121247462206,
                3.9961464788583529,
                19.217858603604583,
                8.7782878752537830,
                5.0038535211416448,
            ),
        ),
    ]
    for m, lrt, x2, df, f1 in cases:  # loglin(t3, m, fit = TRUE)
        r = loglinear_three_way(t, m)
        assert close(r["lrt"], lrt, 1e-9) and close(r["pearson"], x2, 1e-9) and r["df"] == df
        got = [r["fit"][0][j][k] for k in range(2) for j in range(3)]
        assert all(close(a, b, 1e-9) for a, b in zip(got, f1))
    r = loglinear_three_way(t, [(0, 1), (2,)])
    assert close(r["residuals"][1][2][0], 0.2668000487440963, 1e-9)
    # book Table 8.8 (drug example): model A X^2 1410.98 (from rounded fits), model D G^2 0.374 on 1 df
    y = [911, 538, 44, 456, 3, 43, 2, 279]
    d = [[[y[i + 2 * j + 4 * k] for k in range(2)] for j in range(2)] for i in range(2)]
    r = loglinear_three_way(d, [(0,), (1,), (2,)])
    assert (
        close(r["lrt"], 1286.0199544236668, 1e-10) and close(r["pearson"], 1411.3860253098717, 1e-10) and r["df"] == 4
    )
    r = loglinear_three_way(d, [(0, 1), (0, 2), (1, 2)])
    assert close(r["lrt"], 0.37398587014248030, 1e-9) and r["df"] == 1
    assert close(r["fit"][0][0][0], 910.38316965272815651, 1e-10)


def test_linear_system():
    r = solve_linear_system([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
    assert r["kind"] == "unique" and all(close(a, b) for a, b in zip(r["x"], (2, 3, -1)))
    r = solve_linear_system([[1, t] for t in range(1, 6)], [2.1, 3.9, 6.2, 7.8, 10.1])
    assert r["kind"] == "least-squares" and not r["consistent"]
    assert close(r["x"][0], 0.050000000000002619, 1e-11) and close(r["x"][1], 1.989999999999999769)
    r = solve_linear_system([[1, 2], [2, 4]], [1, 3])
    assert (r["rank_A"], r["rank_Ab"], r["kind"], r["x"]) == (1, 2, "rank-deficient", None)
    assert solve_linear_system([[1, 2], [2, 4]], [1, 2])["consistent"]
