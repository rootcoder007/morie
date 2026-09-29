"""Tests for defint.definite_integral: FTC against closed forms."""

import math

from morie.fn.defint import definite_integral


def test_closed_forms():
    r = definite_integral("x*exp(2*x)", "x", 0.0, 1.0)
    assert r["method"] == "FTC"
    assert abs(r["estimate"] - (math.exp(2) + 1) / 4) < 1e-12
    assert abs(r["numeric"] - r["estimate"]) < 1e-10
    assert abs(definite_integral("sin(x)", "x", 0.0, math.pi)["estimate"] - 2.0) < 1e-12
    assert abs(definite_integral("1/(x^2 + 1)", "x", -1.0, 1.0)["estimate"] - math.pi / 2) < 1e-12
