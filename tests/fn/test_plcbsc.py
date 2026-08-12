"""Tests for plcbsc (Abadie, Diamond & Hainmueller 2015)."""

import math

from morie.fn.plcbsc import (in_time_placebo, placebo_inference, plcbsc,
                             simplex_project, synthetic_control)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _panel(T=20, t0=12, J=15, effect=3.0, seed=17):
    rnd = _lcg(seed)
    factor = [math.sin(t / 3.0) for t in range(T)]
    donors = []
    for _ in range(J):
        load = 0.5 + rnd()
        base = 5.0 * rnd()
        donors.append([base + load * factor[t] + 0.05 * (rnd() - 0.5)
                       for t in range(T)])
    mix = [0.5, 0.3, 0.2] + [0.0] * (J - 3)
    y1 = [sum(donors[j][t] * mix[j] for j in range(J)) +
          (effect if t >= t0 else 0.0) for t in range(T)]
    return y1, donors, t0


def test_simplex_projection():
    p = simplex_project([2.0, 0.0, -1.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert all(v >= 0 for v in p) and abs(p[0] - 1.0) < 1e-12
    q = [0.2, 0.3, 0.5]
    assert all(abs(a - b) < 1e-12 for a, b in zip(simplex_project(q), q))


def test_weights_live_on_the_simplex_and_fit_a_convex_target():
    rnd = _lcg(3)
    D = [[rnd() * 10 for _ in range(12)] for _ in range(4)]
    true_w = [0.5, 0.3, 0.2, 0.0]
    x1 = [sum(D[j][i] * true_w[j] for j in range(4)) for i in range(12)]
    fit = synthetic_control(x1, D)
    assert all(w >= -1e-12 for w in fit["weights"])
    assert abs(sum(fit["weights"]) - 1.0) < 1e-8
    assert fit["loss"] < 1e-6
    # a target outside the donors' hull cannot be reached, and the weights
    # still stay on the simplex rather than extrapolating
    out = synthetic_control([30.0] * 12, D)
    assert out["loss"] > 1.0
    assert all(-1e-12 <= w <= 1 + 1e-12 for w in out["weights"])


def test_effect_and_permutation_pvalue():
    y1, donors, t0 = _panel()
    r = plcbsc(y1, donors, t0)
    assert abs(r["estimate"] - 3.0) < 0.3
    assert r["rmspe_pre"] < 0.1
    stats = [abs(r["estimate"])] + [abs(v) for v in r["placebo"]]
    want = (sum(1 for s in stats if s >= abs(r["estimate"]) - 1e-12) /
            float(len(stats)))
    assert abs(r["pvalue"] - want) < 1e-12
    assert r["pvalue"] >= 1.0 / (len(donors) + 1) - 1e-12
    assert r["rank"] == 1
    assert len(r["placebo"]) == len(donors)


def test_no_effect_gives_no_significance():
    y1, donors, t0 = _panel(effect=0.0)
    r = plcbsc(y1, donors, t0)
    assert abs(r["estimate"]) < 0.2
    assert r["pvalue"] > 3.0 / (len(donors) + 1)


def test_rmspe_ratio_statistic():
    y1, donors, t0 = _panel()
    r = plcbsc(y1, donors, t0, statistic="rmspe_ratio")
    assert abs(r["estimate"] - r["rmspe_post"] / r["rmspe_pre"]) < 1e-9
    assert r["rank"] == 1


def test_in_time_placebo():
    y1, donors, t0 = _panel()
    it = in_time_placebo(y1, donors, t0, t0 - 5)
    assert abs(it["placebo_effect"]) < 0.3
    assert abs(sum(it["gaps"][t0:]) / (len(y1) - t0)) > 2.0


def test_validation():
    y1, donors, t0 = _panel()
    for call in (lambda: plcbsc(y1, [], t0),
                 lambda: plcbsc(y1, [donors[0][:-1]], t0),
                 lambda: plcbsc(y1, donors, 0),
                 lambda: plcbsc(y1, donors, t0, statistic="tstat"),
                 lambda: in_time_placebo(y1, donors, t0, t0 + 1),
                 lambda: synthetic_control([1.0, 2.0], [[1.0]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert placebo_inference is plcbsc
