"""Survey-weighted Cox regression with Binder's variance."""
import importlib
import math

import pytest

V = importlib.import_module("morie.fn.svycox")
CX = importlib.import_module("morie.fn.coxph")

T = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
E = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
X = [[0.0], [2.0], [1.0], [1.5], [0.0], [1.0],
     [0.3], [1.0], [0.0], [2.5], [0.7], [1.8]]
W = [1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0]


def test_unit_weights_reproduce_the_unweighted_cox_fit():
    a, b = V.svycoxph(T, E, X), CX.coxph(T, E, X)
    assert a["coefficients"][0] == pytest.approx(b["coefficients"][0],
                                                 abs=1e-10)
    assert a["model_std_errors"][0] == pytest.approx(b["se"][0],
                                                     abs=1e-8)


def test_ties_diverge_from_efron_as_documented():
    tt = [1.0, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    te = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    tx = [[0.0], [2.0], [1.0], [1.0], [0.0], [1.0], [0.0], [1.0],
          [0.0], [2.0]]
    a, b = V.svycoxph(tt, te, tx), CX.coxph(tt, te, tx)
    assert a["ties"] == "breslow"
    assert abs(a["coefficients"][0] - b["coefficients"][0]) > 1e-6


def test_integer_weights_equal_replicated_rows():
    rt, re, rx = [], [], []
    for i in range(len(T)):
        for _ in range(int(W[i])):
            rt.append(T[i])
            re.append(E[i])
            rx.append(X[i])
    assert V.svycoxph(T, E, X, W)["coefficients"][0] \
        == pytest.approx(V.svycoxph(rt, re, rx)["coefficients"][0],
                         abs=1e-9)


@pytest.mark.parametrize("c", [0.5, 10.0, 1000.0])
def test_scaling_all_weights_leaves_beta_alone(c):
    assert V.svycoxph(T, E, X, [c] * len(T))["coefficients"][0] \
        == pytest.approx(V.svycoxph(T, E, X)["coefficients"][0],
                         abs=1e-9)


def test_the_score_vanishes_at_the_estimate():
    r = V.svycoxph(T, E, X, W)
    assert all(abs(u) < 1e-7 for u in r["score"])


def test_the_score_residuals_are_not_trivial():
    r0 = V.score_residuals(T, E, X, [0.0], W)
    assert len(r0) == len(T)
    assert abs(sum(W[i] * r0[i][0] for i in range(len(T)))) > 1e-6


def _clustered(shared):
    t, e, x, w, cl = [], [], [], [], []
    for g in range(12):
        for k in range(4):
            xv = ((g % 2) * 1.0 if shared
                  else ((g * 4 + k) % 2) * 1.0)
            t.append(1.0 + (g * 4 + k) * 0.37)
            e.append(1 if (g * 4 + k) % 5 else 0)
            x.append([xv])
            w.append(25.0)
            cl.append("g%d" % g)
    return t, e, x, w, cl


def test_clustering_inflates_the_design_based_error():
    ti, ei, xi, wi, ci = _clustered(False)
    tc, ec, xc, wc, cc = _clustered(True)
    ind = V.svycoxph(ti, ei, xi, wi, None, ci)
    clu = V.svycoxph(tc, ec, xc, wc, None, cc)
    assert clu["std_errors"][0] > 1.5 * ind["std_errors"][0]
    assert clu["design_effect"][0] > 1.5


def test_ignoring_the_design_hides_it_but_not_the_estimate():
    tc, ec, xc, wc, cc = _clustered(True)
    clu = V.svycoxph(tc, ec, xc, wc, None, cc)
    flat = V.svycoxph(tc, ec, xc, wc)
    assert flat["std_errors"][0] < clu["std_errors"][0]
    assert flat["coefficients"][0] == pytest.approx(
        clu["coefficients"][0], abs=1e-12)


ST = [1.0, 1.2, 1.4, 1.6, 1.8, 9.0, 2.0, 8.4, 8.8, 9.2, 9.6, 10.0]
SE = [1] * 12
SX = [[1.0]] * 6 + [[0.0]] * 6


def test_an_earlier_failing_group_gets_a_positive_coefficient():
    r = V.svycoxph(ST, SE, SX)
    assert r["coefficients"][0] > 1.0
    assert r["z"][0] > 2.0


def test_the_hazard_ratio_is_the_exponential():
    r = V.svycoxph(ST, SE, SX)
    assert r["hazard_ratios"][0] == pytest.approx(
        math.exp(r["coefficients"][0]))


def test_reversing_the_grouping_flips_the_sign():
    a = V.svycoxph(ST, SE, SX)["coefficients"][0]
    b = V.svycoxph(ST, SE, [[1.0 - v[0]] for v in SX])["coefficients"][0]
    assert a == pytest.approx(-b, abs=1e-8)


def test_complete_separation_is_reported():
    sep = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 8.0, 8.4, 8.8, 9.2, 9.6,
           10.0]
    with pytest.raises(ValueError, match="separated"):
        V.svycoxph(sep, SE, SX)


@pytest.mark.parametrize("call", [
    lambda: V.svycoxph(T, E[:5], X),
    lambda: V.svycoxph(T, E, X[:5]),
    lambda: V.svycoxph(T, [0] * 12, X),
    lambda: V.svycoxph(T, E, X, [1.0] * 5),
    lambda: V.svycoxph(T, E, X, [0.0] * 12),
    lambda: V.svycoxph(T, [2] * 12, X),
    lambda: V.svycoxph(T, E, [[1.0]] * 12),
    lambda: V.svycoxph(T, E, [[v[0], 2 * v[0]] for v in X]),
    lambda: V.svycoxph(T, E, X, None, ["a"] * 6 + ["b"] * 6,
                       ["c%d" % (i // 6) for i in range(12)]),
    lambda: V.svycoxph([-1.0] + T[1:], E, X),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()


def test_the_entry_point_is_the_fit():
    assert V.survey_cox(T, E, X, W)["coefficients"] \
        == V.svycoxph(T, E, X, W)["coefficients"]
