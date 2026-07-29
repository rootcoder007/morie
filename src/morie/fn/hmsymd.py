# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symbolic differentiation: manipulate algebraic expressions analytically."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_symbolic_diff", "parse", "to_string", "evaluate"]

_FUNCS = ("sin", "cos", "exp", "log", "tanh", "sqrt")


def _tokenize(src):
    out = []
    i = 0
    s = str(src)
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or (c == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            out.append(("num", float(s[i:j])))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            out.append(("name", s[i:j]))
            i = j
        elif c in "+-*/^(),":
            out.append(("op", c))
            i += 1
        else:
            raise ValueError(f"parse: unexpected character {c!r} at position {i}")
    return out


def parse(src):
    """Parse an arithmetic expression into a tuple tree.

    Grammar: ``expr := term (('+'|'-') term)*``,
    ``term := unary (('*'|'/') unary)*``,
    ``unary := '-' unary | power``,
    ``power := atom ['^' unary]`` (right associative),
    ``atom := number | name | name '(' expr ')' | '(' expr ')'``.

    >>> parse("x + 1")
    ('+', ('var', 'x'), ('num', 1.0))
    """
    toks = _tokenize(src)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def eat(kind, val=None):
        nonlocal pos
        t = peek()
        if t is None or t[0] != kind or (val is not None and t[1] != val):
            raise ValueError(f"parse: expected {val or kind} at token {pos}, got {t}")
        pos += 1
        return t

    def expr():
        node = term()
        while peek() and peek()[0] == "op" and peek()[1] in "+-":
            op = eat("op")[1]
            node = (op, node, term())
        return node

    def term():
        node = unary()
        while peek() and peek()[0] == "op" and peek()[1] in "*/":
            op = eat("op")[1]
            node = (op, node, unary())
        return node

    def unary():
        if peek() and peek() == ("op", "-"):
            eat("op", "-")
            return ("neg", unary())
        return power()

    def power():
        base = atom()
        if peek() and peek() == ("op", "^"):
            eat("op", "^")
            return ("^", base, unary())
        return base

    def atom():
        nonlocal pos
        t = peek()
        if t is None:
            raise ValueError("parse: unexpected end of expression")
        if t[0] == "num":
            eat("num")
            return ("num", t[1])
        if t[0] == "name":
            eat("name")
            if peek() == ("op", "("):
                if t[1] not in _FUNCS:
                    raise ValueError(f"parse: unknown function {t[1]!r}; known: {', '.join(_FUNCS)}")
                eat("op", "(")
                arg = expr()
                eat("op", ")")
                return ("call", t[1], arg)
            return ("var", t[1])
        if t == ("op", "("):
            eat("op", "(")
            node = expr()
            eat("op", ")")
            return node
        raise ValueError(f"parse: unexpected token {t}")

    node = expr()
    if pos != len(toks):
        raise ValueError(f"parse: trailing input at token {pos}: {toks[pos:]}")
    return node


def _num(v):
    return ("num", float(v))


def _simplify(t):
    if t[0] in ("num", "var"):
        return t
    if t[0] == "neg":
        a = _simplify(t[1])
        if a[0] == "num":
            return _num(-a[1])
        return ("neg", a)
    if t[0] == "call":
        return ("call", t[1], _simplify(t[2]))
    a, b = _simplify(t[1]), _simplify(t[2])
    op = t[0]
    if a[0] == "num" and b[0] == "num":
        if op == "+":
            return _num(a[1] + b[1])
        if op == "-":
            return _num(a[1] - b[1])
        if op == "*":
            return _num(a[1] * b[1])
        if op == "/" and b[1] != 0:
            return _num(a[1] / b[1])
        if op == "^":
            return _num(a[1] ** b[1])
    if op == "+":
        if a == _num(0):
            return b
        if b == _num(0):
            return a
    if op == "-" and b == _num(0):
        return a
    if op == "*":
        if a == _num(0) or b == _num(0):
            return _num(0)
        if a == _num(1):
            return b
        if b == _num(1):
            return a
    if op == "/":
        if a == _num(0):
            return _num(0)
        if b == _num(1):
            return a
    if op == "^":
        if b == _num(1):
            return a
        if b == _num(0):
            return _num(1)
    return (op, a, b)


def _diff(t, var):
    kind = t[0]
    if kind == "num":
        return _num(0)
    if kind == "var":
        return _num(1) if t[1] == var else _num(0)
    if kind == "neg":
        return ("neg", _diff(t[1], var))
    if kind == "+":
        return ("+", _diff(t[1], var), _diff(t[2], var))
    if kind == "-":
        return ("-", _diff(t[1], var), _diff(t[2], var))
    if kind == "*":
        return ("+", ("*", _diff(t[1], var), t[2]), ("*", t[1], _diff(t[2], var)))
    if kind == "/":
        return ("/", ("-", ("*", _diff(t[1], var), t[2]), ("*", t[1], _diff(t[2], var))), ("^", t[2], _num(2)))
    if kind == "^":
        u, v = t[1], t[2]
        if v[0] == "num":  # power rule
            return ("*", ("*", v, ("^", u, _num(v[1] - 1))), _diff(u, var))
        if u[0] == "num":  # a^u -> a^u ln a u'
            return ("*", ("*", ("^", u, v), ("call", "log", u)), _diff(v, var))
        # general: u^v = exp(v ln u)
        inner = ("*", v, ("call", "log", u))
        return ("*", ("^", u, v), _diff(inner, var))
    if kind == "call":
        f, u = t[1], t[2]
        du = _diff(u, var)
        if f == "sin":
            return ("*", ("call", "cos", u), du)
        if f == "cos":
            return ("neg", ("*", ("call", "sin", u), du))
        if f == "exp":
            return ("*", ("call", "exp", u), du)
        if f == "log":
            return ("/", du, u)
        if f == "tanh":
            return ("*", ("-", _num(1), ("^", ("call", "tanh", u), _num(2))), du)
        if f == "sqrt":
            return ("/", du, ("*", _num(2), ("call", "sqrt", u)))
        raise ValueError(f"geron_symbolic_diff: no derivative rule for {f!r}")
    raise ValueError(f"geron_symbolic_diff: unknown node {t[0]!r}")


def to_string(t):
    """Render a tree back to infix source.

    >>> to_string(parse("2*x + 1"))
    '2 * x + 1'
    """
    k = t[0]
    if k == "num":
        v = t[1]
        return str(int(v)) if float(v).is_integer() else repr(v)
    if k == "var":
        return t[1]
    if k == "neg":
        return f"-{to_string(t[1])}"
    if k == "call":
        return f"{t[1]}({to_string(t[2])})"
    left, right = to_string(t[1]), to_string(t[2])
    if k in "*/^":
        if t[1][0] in ("+", "-"):
            left = f"({left})"
        if t[2][0] in ("+", "-", "*", "/"):
            right = f"({right})"
    return f"{left} {k} {right}"


def evaluate(t, env):
    """Evaluate a tree against ``env`` (a name -> value mapping). No ``eval``."""
    k = t[0]
    if k == "num":
        return float(t[1])
    if k == "var":
        if t[1] not in env:
            raise ValueError(f"evaluate: no value supplied for variable {t[1]!r}")
        return float(env[t[1]])
    if k == "neg":
        return -evaluate(t[1], env)
    if k == "call":
        a = evaluate(t[2], env)
        return {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "tanh": np.tanh, "sqrt": np.sqrt}[t[1]](a)
    a, b = evaluate(t[1], env), evaluate(t[2], env)
    if k == "+":
        return a + b
    if k == "-":
        return a - b
    if k == "*":
        return a * b
    if k == "/":
        if b == 0:
            raise ValueError("evaluate: division by zero")
        return a / b
    if k == "^":
        return a**b
    raise ValueError(f"evaluate: unknown node {k!r}")


def geron_symbolic_diff(expr, var="x", at=None):
    """
    Symbolic differentiation: manipulate algebraic expressions analytically.

    Formula: apply derivative rules symbolically to expression tree

    A real (small) computer-algebra pass: the source is parsed into an
    expression tree, the derivative rules (sum, product, quotient, power,
    chain, and sin/cos/exp/log/tanh/sqrt) are applied to the tree, and the
    result is constant-folded. Symbolic differentiation is *exact* -- no
    step size, no truncation error -- and that is checked, not claimed:
    when `at` is supplied the symbolic derivative is compared against a
    central finite difference and the discrepancy is returned.

    Its known weakness is also visible in the output: repeated product and
    chain rules make the tree grow, which is exactly the expression swell
    that pushed deep learning to reverse-mode autodiff instead.

    Parameters
    ----------
    expr : str or tuple
        Expression source, or an already-parsed tree.
    var : str, default "x"
        Variable to differentiate with respect to.
    at : mapping, optional
        Point at which to evaluate and finite-difference check, e.g.
        ``{"x": 2.0}``.

    Returns
    -------
    result : RichResult
        Keys: derivative, tree, expression, value, numeric_check, error,
        nodes, estimate, n, method.

    Examples
    --------
    >>> r = geron_symbolic_diff("x^2", "x")
    >>> r["derivative"]
    '2 * x'
    >>> geron_symbolic_diff("sin(x)", "x")["derivative"]
    'cos(x)'
    >>> geron_symbolic_diff("3*x + 5", "x")["derivative"]
    '3'

    The chain rule composes, and the answer agrees with a central
    difference at the requested point:

    >>> r2 = geron_symbolic_diff("exp(2*x)", "x", at={"x": 0.5})
    >>> round(float(r2["value"]), 9)
    5.436563657
    >>> bool(r2["error"] < 1e-6)
    True

    A constant with respect to the chosen variable differentiates to 0:

    >>> geron_symbolic_diff("y^3", "x")["derivative"]
    '0'

    References
    ----------
    Géron Appendix A
    """
    tree = parse(expr) if isinstance(expr, str) else expr
    if not (isinstance(tree, tuple) and tree):
        raise ValueError("geron_symbolic_diff: expr must be a source string or an expression tree")
    v = str(var)
    if not v.isidentifier():
        raise ValueError(f"geron_symbolic_diff: {v!r} is not a valid variable name")

    d = _simplify(_diff(tree, v))
    text = to_string(d)

    def _count(t):
        if t[0] in ("num", "var"):
            return 1
        if t[0] == "neg":
            return 1 + _count(t[1])
        if t[0] == "call":
            return 1 + _count(t[2])
        return 1 + _count(t[1]) + _count(t[2])

    value = None
    numeric = None
    err = None
    if at is not None:
        env = dict(at)
        if v not in env:
            raise ValueError(f"geron_symbolic_diff: `at` must supply a value for {v!r}")
        value = float(evaluate(d, env))
        h = 1e-5 * max(1.0, abs(float(env[v])))
        up, dn = dict(env), dict(env)
        up[v] = env[v] + h
        dn[v] = env[v] - h
        numeric = float((evaluate(tree, up) - evaluate(tree, dn)) / (2 * h))
        err = abs(value - numeric)

    return RichResult(
        title="Symbolic differentiation",
        summary_lines=[
            ("d/d" + v, text),
            ("Nodes in derivative", _count(d)),
            ("Value at point", value if value is not None else "n/a"),
        ],
        interpretation=(
            "Symbolic differentiation is exact but the expression grows with every product and chain "
            "rule; that swell is why deep learning uses reverse-mode autodiff on the computation graph instead."
        ),
        payload={
            "derivative": text,
            "tree": d,
            "expression": to_string(tree),
            "value": value,
            "numeric_check": numeric,
            "error": err,
            "nodes": int(_count(d)),
            "var": v,
            "estimate": float(value) if value is not None else float(_count(d)),
            "n": int(_count(tree)),
            "method": "Rule-based differentiation of the parsed expression tree with constant folding",
        },
    )


def cheatsheet():
    return "hmsymd: Symbolic differentiation: manipulate algebraic expressions analytically"
