import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.research_feedback_loop import (
    feedback_loop_bound,
    feedback_loop_limit,
    feedback_loop_meanfield,
    feedback_loop_sim,
    feedback_loop_urn_law,
)


def test_meanfield_recursion_and_closed_forms():
    mf = feedback_loop_meanfield(0.31, 0.22, 7, 13, 300, "naive", 0.15)
    ca, cb = 7.0, 13.0
    pr = 0.31 / 0.53
    for t in range(300):
        x = ca / (ca + cb)
        ca, cb = ca + 0.85 * 0.31 * x + 0.15 * 0.53 * pr, cb + 0.85 * 0.22 * (1 - x) + 0.15 * 0.53 * (1 - pr)
    assert mf["c_a"][-1] == pytest.approx(ca, rel=1e-13) and mf["c_b"][-1] == pytest.approx(cb, rel=1e-13)
    assert max(mf["share_a"]) <= mf["limit"].cap + 1e-12  # rho_cap
    cf = feedback_loop_meanfield(0.3, 0.2, 1, 99, 1000, "corrected")
    assert cf["share_a"][-1] == pytest.approx((1 + 1000 * 0.3) / (100 + 1000 * 0.5), abs=1e-12)
    naive = feedback_loop_meanfield(0.3, 0.2, 10, 10, 500)["share_a"]
    assert all(b > a for a, b in zip(naive, naive[1:]))  # naive_share_increasing


def test_limits():
    assert feedback_loop_limit(0.3, 0.2, 10, 10).share_a == 1.0
    assert feedback_loop_limit(0.2, 0.3, 10, 10).share_a == 0.0
    assert feedback_loop_limit(0.3, 0.2, 10, 10, "corrected").share_a == pytest.approx(0.6, abs=1e-15)
    assert feedback_loop_limit(0.3, 0.3, 2, 6).share_a == 0.25
    assert feedback_loop_limit(0.3, 0.2, 10, 10, rho=0.5).cap == pytest.approx(0.3 / 0.4, abs=1e-15)
    with pytest.raises(ValueError, match="rho"):
        feedback_loop_limit(0.3, 0.2, 1, 1, rho=2)
    with pytest.raises(ValueError, match="single non-missing number"):
        feedback_loop_limit("a", 0.2, 1, 1)


def test_urn_law_is_uniform_with_middle_mass_at_least_a_quarter():
    for n in (2, 3, 10, 17):
        law = feedback_loop_urn_law(n)
        assert law["prob"] == [1 / (n + 1)] * (n + 1)
        mid = sum(1 for j in range(n + 1) if 0.25 <= (1 + j) / (n + 2) <= 0.75) / (n + 1)
        assert law["prob_middle"] == pytest.approx(mid, abs=1e-15) and law["prob_middle"] >= 0.25


def test_rate_bound_is_the_harmonic_sum_and_bounds_the_recursion():
    b = feedback_loop_bound(0.21, 0.2, 1, 99, 2000)
    assert b.bound == pytest.approx(0.01 / 4 * math.fsum(1 / (100 + k * 0.2) for k in range(2000)), rel=1e-14)
    mf = feedback_loop_meanfield(0.21, 0.2, 1, 99, 2000)
    assert mf["share_a"][-1] - mf["share_a"][0] <= b.bound
    with pytest.raises(ValueError, match="lam_a > lam_b"):
        feedback_loop_bound(0.2, 0.3, 1, 1)


def test_sim_replays_the_philox_stream_step_by_step():
    s = feedback_loop_sim(0.3, 0.2, 10, 10, n_steps=200, n_sims=2, seed=7)
    for i in range(2):
        u = random_uniform(400, seed=7, stream=i)
        ca, cb = 10.0, 10.0
        for t in range(200):
            x = ca / (ca + cb)
            visit_a = u[2 * t] < x
            if u[2 * t + 1] < (0.3 if visit_a else 0.2):
                if visit_a:
                    ca += 1
                else:
                    cb += 1
            assert s.share_a[i][t + 1] == ca / (ca + cb)
    r = feedback_loop_sim(0.6, 0.25, 2, 5, n_steps=100, update="corrected", rho=0.3, n_sims=1, seed=11)
    u = random_uniform(300, seed=11, stream=0)
    ca, cb = 2.0, 5.0
    for t in range(100):
        x = ca / (ca + cb)
        if u[3 * t] < 0.3:
            if u[3 * t + 1] < 0.6 / 0.85:
                ca += 1
            else:
                cb += 1
        else:
            visit_a = u[3 * t + 1] < x
            if u[3 * t + 2] < (0.6 if visit_a else 0.25):
                if visit_a:
                    ca += 1 / x
                else:
                    cb += 1 / (1 - x)
    assert r.final[0] == pytest.approx(ca / (ca + cb), abs=1e-15)
    with pytest.raises(ValueError, match="<= 1"):
        feedback_loop_sim(1.5, 0.2, 1, 1)


def test_polya_urn_sim_lands_on_the_uniform_grid():
    s = feedback_loop_sim(1, 1, 1, 1, n_steps=20, n_sims=400, seed=3)
    js = [round(v * 22 - 1) for v in s.final]
    assert all(0 <= j <= 20 and abs((1 + j) / 22 - v) < 1e-12 for j, v in zip(js, s.final))
    freq = [js.count(j) / 400 for j in range(21)]
    assert max(abs(f - 1 / 21) for f in freq) < 0.05
