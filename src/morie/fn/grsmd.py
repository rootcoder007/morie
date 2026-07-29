# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symbolic differentiation of expression trees."""

import numbers

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_symbolic_differentiation"]

_METHOD = "Symbolic differentiation (sum, product, quotient, chain rules)"

_BINARY = {"+", "-", "*", "/", "^"}
_UNARY = {"sin", "cos", "exp", "log", "neg"}


def _is_const(e):
    return isinstance(e, numbers.Number)


def _simplify(e):
    """Constant-fold and drop the 0/1 identities that make output unreadable."""
    if _is_const(e) or isinstance(e, str):
        return e
    op = e[0]
    args = [_simplify(a) for a in e[1:]]
    if all(_is_const(a) for a in args):
        try:
            return _evaluate((op, *args), {})
        except ValueError:
            return (op, *args)
    if op == "+":
        a, b = args
        if a == 0:
            return b
        if b == 0:
            return a
    elif op == "-":
        a, b = args
        if b == 0:
            return a
    elif op == "*":
        a, b = args
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
    elif op == "/":
        a, b = args
        if a == 0:
            return 0
        if b == 1:
            return a
    elif op == "^":
        a, b = args
        if b == 0:
            return 1
        if b == 1:
            return a
    return (op, *args)


def _evaluate(e, env):
    if _is_const(e):
        return float(e)
    if isinstance(e, str):
        if e not in env:
            raise ValueError(f"variable {e!r} has no value in the supplied environment.")
        return float(env[e])
    op = e[0]
    if op in _UNARY:
        v = _evaluate(e[1], env)
        if op == "sin":
            return float(np.sin(v))
        if op == "cos":
            return float(np.cos(v))
        if op == "exp":
            return float(np.exp(v))
        if op == "neg":
            return -v
        if v <= 0:
            raise ValueError(f"log is undefined at {v}.")
        return float(np.log(v))
    a = _evaluate(e[1], env)
    b = _evaluate(e[2], env)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ValueError("division by zero while evaluating the expression.")
        return a / b
    return float(a**b)


def _diff(e, var):
    if _is_const(e):
        return 0
    if isinstance(e, str):
        return 1 if e == var else 0
    op = e[0]
    if op == "+":
        return ("+", _diff(e[1], var), _diff(e[2], var))
    if op == "-":
        return ("-", _diff(e[1], var), _diff(e[2], var))
    if op == "*":                                        # product rule
        return ("+", ("*", _diff(e[1], var), e[2]), ("*", e[1], _diff(e[2], var)))
    if op == "/":                                        # quotient rule
        num = ("-", ("*", _diff(e[1], var), e[2]), ("*", e[1], _diff(e[2], var)))
        return ("/", num, ("^", e[2], 2))
    if op == "^":                                        # power rule (constant exponent)
        if not _is_const(e[2]):
            raise ValueError(
                "only constant exponents are supported; rewrite a^b as exp(b*log(a))."
            )
        return ("*", ("*", e[2], ("^", e[1], e[2] - 1)), _diff(e[1], var))
    if op == "neg":
        return ("neg", _diff(e[1], var))
    if op in _UNARY:                                     # chain rule
        inner = _diff(e[1], var)
        if op == "sin":
            outer = ("cos", e[1])
        elif op == "cos":
            outer = ("neg", ("sin", e[1]))
        elif op == "exp":
            outer = ("exp", e[1])
        else:
            outer = ("/", 1, e[1])
        return ("*", outer, inner)
    raise ValueError(f"unknown operator {op!r}; supported: {sorted(_BINARY | _UNARY)}.")


def _to_str(e):
    if _is_const(e):
        return repr(e)
    if isinstance(e, str):
        return e
    op = e[0]
    if op == "neg":
        return f"-({_to_str(e[1])})"
    if op in _UNARY:
        return f"{op}({_to_str(e[1])})"
    return f"({_to_str(e[1])} {op} {_to_str(e[2])})"


def geron_symbolic_differentiation(expression, var="x", at=None):
    r"""Differentiate an expression tree by applying the rules.

    .. math::
        \frac{d}{dx}[f+g] = f'+g', \qquad
        \frac{d}{dx}[fg] = f'g + fg', \qquad
        \frac{d}{dx}f(g(x)) = f'(g)\,g'(x)

    Symbolic differentiation returns an *expression*, not a number, so it
    is exact and reusable -- and it is also where expression swell comes
    from: the product rule doubles the tree at every application, which is
    why nobody differentiates a neural network this way.  Constant
    folding and the 0/1 identities are applied afterwards to keep the
    result readable, which is cosmetic, not corrective.

    Expressions are nested tuples: ``("*", "x", 3)``, ``("sin", "x")``,
    with strings for variables and numbers for constants.

    Parameters
    ----------
    expression : tuple, str or number
    var : str, optional
        Variable to differentiate with respect to.
    at : mapping, optional
        Variable values; when given, the derivative is also evaluated.

    Returns
    -------
    RichResult
        Payload keys ``derivative`` (tree), ``derivative_str``,
        ``value`` (when ``at`` is given), ``expression_str``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Appendix A, Symbolic Differentiation section.

    Examples
    --------
    Géron's example :math:`f(x) = x^2 y + y + 2`, differentiated in
    ``x``, gives :math:`2xy`:

    >>> f = ("+", ("+", ("*", ("^", "x", 2), "y"), "y"), 2)
    >>> r = geron_symbolic_differentiation(f, "x")
    >>> r["derivative_str"]
    '((2 * x) * y)'
    >>> r = geron_symbolic_differentiation(f, "x", at={"x": 3, "y": 4})
    >>> r["value"]
    24.0

    Chain rule: :math:`d/dx \sin(x^2) = \cos(x^2)\cdot 2x`, which at
    ``x = 0`` is 0:

    >>> c = geron_symbolic_differentiation(("sin", ("^", "x", 2)), "x", at={"x": 0})
    >>> c["derivative_str"]
    '(cos((x ^ 2)) * (2 * x))'
    >>> c["value"]
    0.0
    """
    if not isinstance(var, str) or not var:
        raise ValueError(f"var must be a non-empty variable name, got {var!r}.")

    def _check(e):
        if _is_const(e) or isinstance(e, str):
            return
        if not isinstance(e, tuple) or len(e) < 2:
            raise ValueError(f"malformed sub-expression {e!r}; expected (op, arg[, arg]).")
        op = e[0]
        if op in _UNARY:
            if len(e) != 2:
                raise ValueError(f"{op!r} takes one argument, got {len(e) - 1}.")
        elif op in _BINARY:
            if len(e) != 3:
                raise ValueError(f"{op!r} takes two arguments, got {len(e) - 1}.")
        else:
            raise ValueError(f"unknown operator {op!r}; supported: {sorted(_BINARY | _UNARY)}.")
        for a in e[1:]:
            _check(a)

    _check(expression)
    d = _simplify(_diff(expression, var))
    payload = {
        "derivative": d,
        "derivative_str": _to_str(d),
        "expression_str": _to_str(expression),
        "variable": var,
        "estimate": _to_str(d),
        "n": 1,
        "method": _METHOD,
    }
    lines = [("d/d" + var, _to_str(d))]
    if at is not None:
        if not isinstance(at, dict):
            raise ValueError("at must be a mapping from variable name to value.")
        val = _evaluate(d, at)
        payload["value"] = val
        payload["estimate"] = val
        lines.append(("Value", val))

    return RichResult(title="Symbolic differentiation", summary_lines=lines, payload=payload)


def cheatsheet():
    return "grsmd: trees like ('*','x',3); sum/product/quotient/power/chain rules, then constant folding"
