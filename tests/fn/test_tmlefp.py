"""Tests for tmlefp (Crump, Hotz, Imbens & Mitnik 2009)."""

import math

from morie.fn.tmlefp import (alpha_from_gamma, optimal_alpha,
                             optimal_alpha_att, optimal_overlap,
                             owate_weights, tmlefp)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _bad_overlap(n=400, seed=11):
    rnd = _lcg(seed)
    out = []
    for _ in range(n):
        u = rnd()
        out.append(0.01 + 0.98 * u ** 3 if u < 0.5 else 0.02 + 0.96 * u)
    return out


def test_alpha_from_gamma_inverts_the_threshold():
    for g in (4.0, 5.0, 100.0, 1e6):
        a = alpha_from_gamma(g)
        assert abs(1.0 / (a * (1.0 - a)) - g) < 1e-6 * g
        assert 0.0 < a <= 0.5 + 1e-12
    assert alpha_from_gamma(4.0) == 0.5
    try:
        alpha_from_gamma(3.9)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_no_trimming_when_sup_k_is_small():
    r = optimal_alpha([0.5] * 100)
    assert r["no_trimming"] and r["trim"] == 0


def test_fixed_point_solves_its_own_equation():
    e = _bad_overlap()
    r = optimal_alpha(e)
    assert not r["no_trimming"]
    k = [1.0 / (v * (1 - v)) for v in e]
    sel = [v for v in k if v < r["gamma"]]
    assert abs(r["gamma"] - 2.0 * sum(sel) / len(sel)) < 1e-6
    assert abs(1.0 / (r["alpha"] * (1 - r["alpha"])) - r["gamma"]) < 1e-6
    assert all(r["keep"][i] ==
               (r["alpha"] - 1e-12 <= e[i] <= 1 - r["alpha"] + 1e-12)
               for i in range(len(e)))


def test_trimming_lowers_the_variance_bound():
    e = _bad_overlap()
    r = tmlefp([0.0] * len(e), [i % 2 for i in range(len(e))], e)
    assert r["variance_bound"] < r["variance_bound_full"]
    assert r["n_kept"] + r["n_trimmed"] == r["n"]


def test_heteroskedastic_rule():
    e = _bad_overlap()
    n = len(e)
    same = optimal_alpha(e, [2.0] * n, [2.0] * n)
    assert same["keep"] == optimal_alpha(e)["keep"]
    lop = optimal_alpha(e, [5.0] * n, [0.2] * n)
    k = [5.0 / v + 0.2 / (1 - v) for v in e]
    sel = [v for v in k if v < lop["gamma"]]
    assert abs(lop["gamma"] - 2.0 * sum(sel) / len(sel)) < 1e-6
    assert lop["alpha"] != lop["alpha"]      # NaN: not an interval in e


def test_att_rule_is_one_sided():
    e = _bad_overlap()
    rnd = _lcg(5)
    w = [1 if v > rnd() else 0 for v in e]
    att = optimal_alpha_att(e, w)
    assert all(att["keep"][i] == (e[i] <= att["alpha_t"])
               for i in range(len(e)))
    g = [1.0 / (1.0 - e[i]) for i in range(len(e))
         if w[i] == 1 and e[i] <= att["alpha_t"]]
    assert abs(1.0 / (1.0 - att["alpha_t"]) -
               2.0 * sum(g) / len(g)) < 1e-6


def test_owate_weights():
    e = _bad_overlap()
    om = owate_weights(e)
    assert all(abs(om[i] - e[i] * (1 - e[i])) < 1e-15
               for i in range(len(e)))
    het = owate_weights(e, [3.0] * len(e), [1.0] * len(e))
    assert all(abs(het[i] - 1.0 / (3.0 / e[i] + 1.0 / (1 - e[i]))) < 1e-15
               for i in range(len(e)))


def test_recovers_a_known_effect():
    rnd = _lcg(99)
    Y, W, E = [], [], []
    for _ in range(3000):
        x = rnd()
        e = 0.05 + 0.9 * x
        t = 1 if rnd() < e else 0
        u1 = max(rnd(), 1e-12)
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * rnd())
        Y.append(1.0 * x + 2.0 * t + 0.3 * z)
        W.append(t)
        E.append(e)
    r = tmlefp(Y, W, E)
    assert abs(r["estimate"] - 2.0) < 0.2
    assert abs(r["owate"] - 2.0) < 0.2
    assert abs(r["ate_full"] - 2.0) < 0.2


def test_validation():
    for call in (lambda: optimal_alpha([0.0, 0.5]),
                 lambda: optimal_alpha([1.0, 0.5]),
                 lambda: optimal_alpha([]),
                 lambda: tmlefp([1.0, 2.0], [0, 2], [0.4, 0.6]),
                 lambda: tmlefp([1.0], [0, 1], [0.4, 0.6]),
                 lambda: tmlefp([1.0, 2.0], [0, 1], [0.4, 0.6],
                                estimand="atu"),
                 lambda: optimal_alpha_att([0.4, 0.6], [0, 0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert optimal_overlap is tmlefp
