"""Verification tests for information_theory_mackay19e14.sexfsol.

The expected values are recomputed from MacKay (2003) eq. (19.14) p. 273 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay19e14 import sexfsol


def test_sexfsol_satisfies_its_own_initial_condition():
    g, f0 = 64.0, 0.2
    res = sexfsol(0.0, g, f0)
    assert res["f"] == pytest.approx(f0, abs=1e-12)


def test_sexfsol_reports_the_printed_constant_alongside_the_consistent_one():
    g, f0 = 64.0, 0.2
    res = sexfsol(0.0, g, f0)
    # the book prints c = asin(2 f0 - 1), which does not satisfy f(0) = f0
    assert res["cbook"] == pytest.approx(math.asin(2.0 * f0 - 1.0), rel=1e-12)
    eta = math.sqrt(2.0 / (math.pi + 2.0))
    assert res["c"] == pytest.approx(math.sqrt(g) / eta * res["cbook"], rel=1e-12)


def test_sexfsol_rate_is_eta_root_f_one_minus_f_over_g():
    """(19.14) differentiates to eta sqrt(f (1-f) / G), not the printed
    (19.13) rate eta sqrt(f (1-f) G).

    The two printed equations disagree by exactly a factor of G: with
    theta = eta (t + c)/sqrt(G), f = (1 + sin theta)/2 gives
    df/dt = (eta/(2 sqrt(G))) cos theta, while f (1 - f) = cos^2(theta)/4
    makes the printed rate (eta sqrt(G)/2) cos theta. Each module
    reproduces its own printed equation, so this test pins the
    derivative of the closed form rather than asserting the pair are
    consistent.
    """
    eta = math.sqrt(2.0 / (math.pi + 2.0))
    h = 1e-6
    for g in (4.0, 36.0, 100.0):
        t, f0 = 1.0, 0.35
        f_here = sexfsol(t, g, f0)["f"]
        slope = (sexfsol(t + h, g, f0)["f"] - sexfsol(t - h, g, f0)["f"]) / (2.0 * h)
        assert slope == pytest.approx(eta * math.sqrt(f_here * (1.0 - f_here) / g), rel=1e-6)


def test_sexfsol_reaches_fixation_at_the_quoted_time():
    g, f0 = 64.0, 0.5
    res = sexfsol(0.0, g, f0)
    # t_perfect = (pi/eta) sqrt(G) carries f from 1/2 to 1
    at_end = sexfsol(res["tperfect"] / 2.0, g, f0)["f"]
    assert at_end == pytest.approx(1.0, abs=1e-9)
