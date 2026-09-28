"""staffing: Erlang identities, brute-force shift schedules, GLM recovery of generating parameters."""

import itertools
import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.staffing import (
    call_demand_model,
    erlang_c,
    patrol_car_requirement,
    relief_factor,
    shift_schedule,
    staffing_requirement,
    workload_staffing,
)


def test_erlang_and_staffing():
    # Erlang C by the direct formula
    a, c = 3.3, 5
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = a**c / math.factorial(c) * c / (c - a)
    assert erlang_c(c, a) == pytest.approx(top / (s + top), abs=1e-14)
    assert math.isnan(erlang_c(2, 2.5))
    r = staffing_requirement([0.5, 2.0, 6.0], 0.75, target="service_level", level=0.9, threshold=0.25, beta=1.0)
    for lam, u in zip([0.5, 2.0, 6.0], r.units):
        a = lam * 0.75
        sl = lambda k: 1 - erlang_c(k, a) * math.exp(-(k - a) * 0.25 / 0.75)  # noqa: E731
        assert sl(u) >= 0.9 and (u - 1 <= a or sl(u - 1) < 0.9)
    assert r.square_root_units == [math.ceil(a + math.sqrt(a)) for a in (0.375, 1.5, 4.5)]
    asa = staffing_requirement([3.0], 0.5, target="asa", level=0.05)
    k = asa.units[0]
    assert erlang_c(k, 1.5) * 0.5 / (k - 1.5) <= 0.05 < erlang_c(k - 1, 1.5) * 0.5 / (k - 2.5)
    p = patrol_car_requirement(
        3.0,
        0.5,
        area=25.0,
        response_speed=30.0,
        target_response=0.1,
        street_miles=60.0,
        patrol_speed=15.0,
        patrol_frequency=0.5,
    )
    assert p.response == math.ceil(1.5 + 25 * (0.5 / 3.0) ** 2) and p.patrol == math.ceil(1.5 + 2.0)
    assert p.units == max(p.queue, p.response, p.patrol)


def test_shift_schedule_is_optimal():
    for req, L in (([2, 2, 3, 5, 5, 4], 3), ([1, 4, 2, 0, 3, 3, 1, 2], 3), ([3, 1, 1, 4, 2], 2)):
        r = shift_schedule(req, L)
        n = len(req)
        assert all(r.coverage[h] >= req[h] for h in range(n))
        assert r.coverage == [sum(r.starts[(h - k) % n] for k in range(L)) for h in range(n)]
        best = min(
            sum(x)
            for x in itertools.product(range(max(req) + 1), repeat=n)
            if all(sum(x[(h - k) % n] for k in range(L)) >= req[h] for h in range(n))
        )
        assert r.total == best
    with pytest.raises(ValueError):
        shift_schedule([1, 2], 3)


def test_demand_model_relief_workload():
    t = list(range(24 * 7 * 6))
    u = random_uniform(len(t), seed=3, stream=0)
    true = [math.exp(1.2 + 0.5 * math.sin(2 * math.pi * h / 24) - 0.3 * math.cos(2 * math.pi * h / 168)) for h in t]
    y = []
    for ui, m in zip(u, true):  # Poisson draw by inversion
        k, p = 0, math.exp(-m)
        cdf = p
        while float(ui) > cdf:
            k += 1
            p *= m / k
            cdf += p
        y.append(k)
    r = call_demand_model(y, t, daily=1, weekly=1, annual=0, trend=False, new_hours=[1000.0])
    b = r.coefficients
    assert abs(b[0] - 1.2) < 4 * r.se[0] and abs(b[1] - 0.5) < 4 * r.se[1] and abs(b[4] + 0.3) < 4 * r.se[4]
    # score equations hold at the MLE: X'(y - mu) = 0
    X = [
        [
            1.0,
            math.sin(2 * math.pi * h / 24),
            math.cos(2 * math.pi * h / 24),
            math.sin(2 * math.pi * h / 168),
            math.cos(2 * math.pi * h / 168),
        ]
        for h in t
    ]
    for j in range(5):
        assert sum(x[j] * (yy - m) for x, yy, m in zip(X, y, r.fitted)) == pytest.approx(0.0, abs=1e-6)
    xn = [
        1.0,
        math.sin(2 * math.pi * 1000 / 24),
        math.cos(2 * math.pi * 1000 / 24),
        math.sin(2 * math.pi * 1000 / 168),
        math.cos(2 * math.pi * 1000 / 168),
    ]
    assert r.forecast[0] == pytest.approx(math.exp(sum(a * c for a, c in zip(b, xn))))
    assert relief_factor(151) == pytest.approx(365 / 214)
    w = workload_staffing([30, 10], [45, 90], units_per_call=[1, 2], call_share=0.5, relief=1.7)
    assert (w.obligated_hours, w.on_duty, w.on_duty_ceiling) == (52.5, 13.125, 14)
