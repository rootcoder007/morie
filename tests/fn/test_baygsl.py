"""Auxiliary-variable slice sampling."""
import importlib
import math

import pytest

B = importlib.import_module("morie.fn.baygsl")

NORMAL = lambda x: -0.5 * x * x
EXPO = lambda x: -x if x > 0 else float("-inf")


def moments(d):
    n = len(d)
    m = sum(d) / n
    return m, sum((t - m) ** 2 for t in d) / (n - 1)


def test_a_normal_target():
    m, v = moments(B.slice_chain(NORMAL, 0.0, n=6000, w=1.0,
                                 burn=500, seed=3)["draws"])
    assert abs(m) < 0.05
    assert abs(v - 1.0) < 0.06


def test_a_bounded_target_stays_bounded():
    r = B.slice_chain(EXPO, 1.0, n=6000, w=1.0, burn=500, seed=5,
                      lower=0.0)
    m, v = moments(r["draws"])
    assert abs(m - 1.0) < 0.06
    assert abs(v - 1.0) < 0.12
    assert min(r["draws"]) > 0.0


def test_a_gamma_target():
    g = lambda x: (2.0 * math.log(x) - 2.0 * x if x > 0
                   else float("-inf"))
    m, v = moments(B.slice_chain(g, 1.0, n=6000, w=1.0, burn=500,
                                 seed=7, lower=0.0)["draws"])
    assert abs(m - 1.5) < 0.06
    assert abs(v - 0.75) < 0.08


@pytest.mark.parametrize("w", [0.05, 0.5, 5.0, 50.0])
def test_the_width_does_not_change_the_answer(w):
    m, v = moments(B.slice_chain(NORMAL, 0.0, n=6000, w=w, burn=500,
                                 seed=11)["draws"])
    assert abs(m) < 0.06
    assert abs(v - 1.0) < 0.07


def test_the_width_does_change_the_cost():
    lo = B.slice_chain(NORMAL, 0.0, n=4000, w=50.0, burn=500,
                       seed=11)["evals_per_draw"]
    hi = B.slice_chain(NORMAL, 0.0, n=4000, w=0.05, burn=500,
                       seed=11)["evals_per_draw"]
    assert hi / lo > 3.0


def test_the_normalising_constant_is_irrelevant():
    a = B.slice_chain(lambda x: NORMAL(x) + 137.0, 0.0, n=2000,
                      w=1.0, burn=200, seed=3)["draws"]
    b = B.slice_chain(NORMAL, 0.0, n=2000, w=1.0, burn=200,
                      seed=3)["draws"]
    assert a == b


def _mix(x, sep, wt):
    a = -0.5 * (x - sep) ** 2
    b = -0.5 * (x + sep) ** 2
    hi = max(a, b)
    return hi + math.log(wt * math.exp(a - hi)
                         + (1 - wt) * math.exp(b - hi))


def test_overlapping_modes_are_crossed():
    r = B.slice_chain(lambda x: _mix(x, 2.0, 0.3), -2.0, n=20000,
                      w=4.0, burn=2000, seed=13)
    right = sum(1 for t in r["draws"] if t > 0) / len(r["draws"])
    assert abs(right - 0.3) < 0.06


def test_separated_modes_are_not():
    # Documented limitation of stepping out, not a defect.
    r = B.slice_chain(lambda x: _mix(x, 6.0, 0.3), -6.0, n=20000,
                      w=4.0, burn=2000, seed=13)
    right = sum(1 for t in r["draws"] if t > 0) / len(r["draws"])
    assert right < 0.001


RHO = 0.8
C = [lambda v, s: -0.5 * (v - RHO * s[1]) ** 2 / (1 - RHO ** 2),
     lambda v, s: -0.5 * (v - RHO * s[0]) ** 2 / (1 - RHO ** 2)]


def test_a_gibbs_sweep_recovers_a_correlated_normal():
    g = B.gibbs_slice(C, [0.0, 0.0], n=8000, w=1.0, burn=1000,
                      seed=17)
    xs = [r[0] for r in g["draws"]]
    ys = [r[1] for r in g["draws"]]
    mx, vx = moments(xs)
    my, vy = moments(ys)
    assert abs(mx) < 0.06 and abs(my) < 0.06
    assert abs(vx - 1.0) < 0.08 and abs(vy - 1.0) < 0.08
    cov = sum((xs[i] - mx) * (ys[i] - my)
              for i in range(len(xs))) / (len(xs) - 1)
    assert abs(cov / math.sqrt(vx * vy) - RHO) < 0.03


def test_the_ess_is_reported_and_bounded():
    g = B.gibbs_slice(C, [0.0, 0.0], n=3000, w=1.0, burn=500,
                      seed=17)
    assert all(0 < e <= len(g["draws"]) for e in g["ess"])


def test_the_entry_point_matches():
    a = B.hybrid_gibbs_slice(C, [0.0, 0.0], n=500, w=1.0, seed=2)
    b = B.gibbs_slice(C, [0.0, 0.0], n=500, w=1.0, seed=2)
    assert a["draws"] == b["draws"]


@pytest.mark.parametrize("call", [
    lambda: B.slice_chain(NORMAL, 0.0, n=100, w=0.0),
    lambda: B.slice_chain(NORMAL, 0.0, n=100, w=-1.0),
    lambda: B.slice_chain(EXPO, -1.0, n=100, w=1.0),
    lambda: B.slice_chain(NORMAL, 0.0, n=0),
    lambda: B.slice_chain(NORMAL, 0.0, n=10, thin=0),
    lambda: B.effective_sample_size([1.0, 2.0]),
    lambda: B.gibbs_slice([C[0]], [0.0, 0.0]),
    lambda: B.gibbs_slice([], []),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
