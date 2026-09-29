# morie.fn -- function file (rootcoder007/morie)
"""Definite integral via FTC."""

import math

from ._richresult import RichResult
from .symint import _ev, _parse, symbolic_integrate


def definite_integral(expr, x="x", a=0.0, b=1.0, n_check=2000):
    r"""Definite integral ``int_a^b f(x) dx = F(b) - F(a)`` by the fundamental theorem of calculus.

    ``expr`` is an elementary expression in ``x`` (``+ - * / ^``, ``sqrt``,
    ``sin cos tan exp log asin acos atan sinh cosh``); its antiderivative
    ``F`` comes from :func:`morie.fn.symint.symbolic_integrate` and the
    integral is ``F(b) - F(a)`` (valid when ``f`` is continuous on
    ``[a, b]``). A composite Simpson rule with ``n_check`` panels is
    reported alongside (``numeric``) as a check; when no antiderivative is
    found the Simpson value is returned with ``method = "Simpson"``.

    References
    ----------
    Apostol, T. M. (1967). *Calculus*, vol. 1, 2nd ed., sec. 5.1 (the first
    and second fundamental theorems). Wiley.

    Examples
    --------
    >>> r = definite_integral("x*exp(2*x)", "x", 0.0, 1.0)
    >>> round(r["estimate"], 12), r["method"]
    (2.097264024733, 'FTC')
    """
    e = _parse(expr)
    a, b = float(a), float(b)
    m = int(n_check) + int(n_check) % 2
    hstep = (b - a) / m
    vals = [_ev(e, {x: a + i * hstep}) for i in range(m + 1)]
    simpson = hstep / 3.0 * (vals[0] + vals[-1] + 4.0 * math.fsum(vals[1:-1:2]) + 2.0 * math.fsum(vals[2:-1:2]))
    F = symbolic_integrate(expr, x)["antiderivative"]
    if F is None:
        return RichResult(
            payload={"estimate": simpson, "antiderivative": None, "numeric": simpson, "method": "Simpson"}
        )
    Fe = _parse(F)
    val = _ev(Fe, {x: b}) - _ev(Fe, {x: a})
    return RichResult(payload={"estimate": val, "antiderivative": F, "numeric": simpson, "method": "FTC"})


def cheatsheet():
    return "defint: int_a^b f = F(b) - F(a) with F from symbolic_integrate (Simpson check)"
