"""Tests for morie.fn.sirepi: RK4 steps recomputed and the final-size relation."""

import math

from morie.fn.sirepi import sir_compartmental


def test_first_rk4_step():
    S0, I0, R0, b, g, h = 990.0, 10.0, 0.0, 0.3, 0.1, 0.5
    N = S0 + I0 + R0

    def f(s, i):
        inf = b * s * i / N
        return -inf, inf - g * i

    k1 = f(S0, I0)
    k2 = f(S0 + h / 2 * k1[0], I0 + h / 2 * k1[1])
    k3 = f(S0 + h / 2 * k2[0], I0 + h / 2 * k2[1])
    k4 = f(S0 + h * k3[0], I0 + h * k3[1])
    r = sir_compartmental(S0, I0, R0, b, g, 1.0, dt=h)
    assert abs(r["S"][1] - (S0 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))) < 1e-10
    assert abs(r["I"][1] - (I0 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))) < 1e-10
    assert all(abs(s + i + rr - N) < 1e-9 for s, i, rr in zip(r["S"], r["I"], r["R"]))


def test_long_run_matches_the_final_size_equation():
    r = sir_compartmental(990, 10, 0, 0.3, 0.1, 400)
    fs = r["final_S"]
    assert abs(fs - 990 * math.exp(-(0.3 / 0.1 / 1000) * (1000 - fs))) < 1e-9
    assert abs(r["S"][-1] - fs) < 1e-6
    assert abs(r["R0_basic"] - 0.3 / 0.1) < 1e-15
