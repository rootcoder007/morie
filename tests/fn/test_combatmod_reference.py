"""combatmod: closed forms against RK4, conservation laws of each Lanchester law, Engel estimates, salvo arithmetic."""

import math

import pytest

from morie.fn.combatmod import (
    blotto_best_response,
    blotto_payoff,
    lanchester_fit,
    lanchester_simulate,
    lanchester_square_outcome,
    salvo_exchange,
)


def test_square_law_closed_form_and_invariant():
    o = lanchester_square_outcome(50.0, 80.0, 0.02, 0.05)  # b x0^2 = 125 > a y0^2 = 128? no: Y wins
    assert o.winner == "y" and o.survivors == pytest.approx(math.sqrt(6400 - 2.5 * 2500))
    g = math.sqrt(0.001)
    assert o.duration == pytest.approx(math.atanh(50 * math.sqrt(2.5) / 80) / g)
    s = lanchester_simulate(50.0, 80.0, 0.02, 0.05, t_end=500.0, dt=0.005)
    assert s.ended_by == "x annihilated" and s.y[-1] == pytest.approx(o.survivors, rel=1e-6)
    assert s.t[-1] == pytest.approx(o.duration, rel=1e-6)
    # state equation b (x0^2 - x^2) = a (y0^2 - y^2) along the path
    for x, y in zip(s.x[::200], s.y[::200]):
        assert 0.05 * (2500 - x * x) == pytest.approx(0.02 * (6400 - y * y), abs=1e-6)
    d = lanchester_square_outcome(10.0, 10.0, 1.0, 1.0)
    assert d.winner == "draw" and d.duration == math.inf


def test_other_laws_conservation():
    lin = lanchester_simulate(100.0, 70.0, 0.002, 0.003, law="linear", t_end=5.0, dt=0.001)
    for x, y in zip(lin.x[::500], lin.y[::500]):
        assert 0.003 * (100 - x) == pytest.approx(0.002 * (70 - y), abs=1e-9)
    mix = lanchester_simulate(40.0, 90.0, 0.001, 0.2, law="mixed", t_end=3.0, dt=0.001)
    for x, y in zip(mix.x[::500], mix.y[::500]):
        # dx/dy = a x y / (b x)  =>  b (x0 - x) = (a / 2)(y0^2 - y^2)
        assert 0.2 * (40 - x) == pytest.approx(0.0005 * (8100 - y * y), abs=1e-7)
    lg = lanchester_simulate(100.0, 50.0, 0.1, 0.2, law="logarithmic", t_end=2.0, dt=0.01)
    assert lg.x[-1] == pytest.approx(100 * math.exp(-0.2), rel=1e-9) and lg.y[-1] == pytest.approx(
        50 * math.exp(-0.4), rel=1e-9
    )
    rf = lanchester_simulate(10.0, 10.0, 0.0, 0.0, t_end=1.0, dt=0.1, reinforce_x=2.0)
    assert rf.x[-1] == pytest.approx(12.0) and rf.y[-1] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        lanchester_simulate(1.0, 1.0, 1.0, 1.0, law="bogus")


def test_fit_recovers_generating_coefficients():
    x, y = [1000.0], [800.0]
    for _ in range(12):
        x.append(x[-1] - 0.03 * y[-1])
        y.append(y[-1] - 0.05 * x[-2])
    f = lanchester_fit(x, y)
    assert (f.a, f.b) == pytest.approx((0.03, 0.05), rel=1e-12)
    assert f.r2_x == pytest.approx(1.0) and f.r2_y == pytest.approx(1.0)
    xp, yp = [1000.0], [800.0]
    for _ in range(12):
        xp.append(xp[-1] - 0.002 * yp[-1] ** 1.2 * xp[-1] ** 0.3)
        yp.append(yp[-1] - 0.004 * xp[-2] ** 0.9 * yp[-1] ** 0.1)
    pw = lanchester_fit(xp, yp, law="power")
    # both forces decline together, so the log-regression columns are nearly collinear: ~1e-8 recovery
    assert (pw.x["a"], pw.x["p"], pw.x["q"]) == pytest.approx((0.002, 1.2, 0.3), rel=1e-6)
    assert (pw.y["a"], pw.y["p"], pw.y["q"]) == pytest.approx((0.004, 0.9, 0.1), rel=1e-6)


def test_salvo_and_blotto():
    s = salvo_exchange(6.0, 4.0, alpha=3.0, beta=4.0, a1=2.0, b1=3.0, a3=1.5, b3=2.0, salvos=2)
    dB = (3 * 6 - 2 * 4) / 3
    dA = (4 * 4 - 1.5 * 6) / 2
    assert s.B[1] == pytest.approx(4 - dB) and s.A[1] == pytest.approx(6 - dA)
    assert s.fractional_exchange_ratio == pytest.approx((dB / 4) / (dA / 6))
    a2, b2 = s.A[1], s.B[1]
    assert s.B[2] == pytest.approx(b2 - min(b2, max(0.0, (3 * a2 - 2 * b2) / 3)))
    over = salvo_exchange(10.0, 1.0, alpha=5.0, beta=0.1, a1=1.0, b1=1.0)
    assert over.B == [1.0, 0.0] and over.A == [10.0, 9.9]
    assert blotto_payoff([4, 1, 1], [2, 2, 2], values=[3, 1, 1]).payoff == 3 - 2
    assert blotto_payoff([2, 2], [2, 2]).payoff == 0.0
    br = blotto_best_response([(0.5, [3, 3, 0]), (0.5, [0, 3, 3])], 6)
    best = max(
        sum(0.5 * blotto_payoff([a, b, 6 - a - b], s)["payoff"] for s in ([3, 3, 0], [0, 3, 3]))
        for a in range(7)
        for b in range(7 - a)
    )
    assert br.expected_payoff == pytest.approx(best)
