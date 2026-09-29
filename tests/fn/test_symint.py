import math

import pytest

from morie.fn.symint import symbolic_diff, symbolic_integrate

NS = {k: getattr(math, k) for k in ("sin", "cos", "tan", "exp", "log", "asin", "acos", "atan", "sinh", "cosh", "sqrt")}


def _f(s, x):
    return eval(s.replace("^", "**"), {"__builtins__": {}}, dict(NS, x=x, pi=math.pi))


CASES = [
    "x*exp(2*x)", "1/(x^2+1)", "2*x*cos(x^2)", "x^3 - 2*x + 5", "sin(3*x+1)", "x^2*sin(x)", "log(x)",
    "1/(x^2-1)", "(x+1)/(x^2+2*x+5)", "x/(x+1)^2", "exp(x)*sin(2*x)", "tan(x)", "atan(x)", "log(x)/x",
    "sqrt(x)", "1/sqrt(4-9*x^2)", "1/sqrt(x^2+2)", "1/(x^3+1)", "x*atan(x)", "x^2*log(3*x)", "1/(x^4-1)",
    "(2*x+3)/(x^2+3*x+2)^2", "cos(x)/sin(x)",
]  # fmt: skip


@pytest.mark.parametrize("expr", CASES)
def test_antiderivative_differentiates_back(expr):
    r = symbolic_integrate(expr)
    assert r.antiderivative is not None and r.verified
    for x in (0.37, 0.61, 1.13):
        h = 1e-5
        try:
            want = _f(expr, x)
            got = (_f(r.antiderivative, x + h) - _f(r.antiderivative, x - h)) / (2 * h)
        except (ValueError, ZeroDivisionError):
            continue
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6)


def test_known_forms():
    assert symbolic_integrate("1/(x^2+1)").antiderivative == "atan(x)"
    assert symbolic_integrate("2*x*cos(x^2)").antiderivative == "sin(x^2)"
    assert symbolic_integrate("exp(x^2)").antiderivative is None
    assert symbolic_integrate("t^2", "t").antiderivative == "0.333333333333333*t^3"


def test_diff():
    d = symbolic_diff("x^3*sin(x)").derivative
    for x in (0.4, 1.7):
        assert _f(d, x) == pytest.approx(3 * x * x * math.sin(x) + x**3 * math.cos(x), abs=1e-12)
    with pytest.raises(ValueError):
        symbolic_integrate("abs(x)")
