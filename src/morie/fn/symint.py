# morie.fn -- function file (rootcoder007/morie)
"""Symbolic integration of elementary expressions: a canonical simplifier, symbolic differentiation, and an
integrator combining an antiderivative table, linearity, rational-function integration by partial fractions
(the rational part of the Risch algorithm), integration by parts and derivative-divides substitution."""

from __future__ import annotations

import math

from ._richresult import RichResult
from .symalg import shunting_yard

__all__ = ["symbolic_integrate", "symbolic_diff"]

_FNS = ("sin", "cos", "tan", "exp", "log", "asin", "acos", "atan", "sinh", "cosh")


# ---------------------------------------------------------------- construction and printing
def _num(v):
    return ("num", float(v))


def _fmt(v):
    if v == 0:
        return "0"
    if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
        return "%d" % int(v)
    return "%.15g" % v


def _prec(e):
    t = e[0]
    if t == "add":
        return 1
    if t == "mul":
        return 2
    if t == "num" and e[1] < 0:
        return 1
    if t == "pow":
        return 3
    return 4


def _str(e):
    t = e[0]
    if t == "num":
        return _fmt(e[1])
    if t == "sym":
        return e[1]
    if t == "fn":
        return f"{e[1]}({_str(e[2])})"
    if t == "pow":
        b, x = e[1], e[2]
        if x == ("num", 0.5):
            return f"sqrt({_str(b)})"
        bs = _str(b) if _prec(b) > 3 else f"({_str(b)})"
        xs = _str(x) if _prec(x) > 3 and not (x[0] == "num" and x[1] < 0) else f"({_str(x)})"
        return f"{bs}^{xs}"
    if t == "mul":
        fs = e[1]
        sign = ""
        parts = []
        rest = fs
        if fs[0][0] == "num":
            c = fs[0][1]
            rest = fs[1:]
            if c < 0:
                sign = "-"
                c = -c
            if c != 1:
                parts.append(_fmt(c))
        for f in rest:
            s = _str(f)
            parts.append(s if _prec(f) > 2 else f"({s})")
        return sign + "*".join(parts)
    # add
    s = _str(e[1][0])
    for term in e[1][1:]:
        ts = _str(term)
        if ts.startswith("-"):
            s += " - " + ts[1:]
        else:
            s += " + " + ts
    return s


def _key(e):
    return _str(e)


# ---------------------------------------------------------------- simplification
def _free(e, x):
    t = e[0]
    if t == "num":
        return True
    if t == "sym":
        return e[1] != x
    if t == "fn":
        return _free(e[2], x)
    if t == "pow":
        return _free(e[1], x) and _free(e[2], x)
    return all(_free(f, x) for f in e[1])


def _isint(v):
    return math.isfinite(v) and v == int(v) and abs(v) <= 64


def _npow(b, n):
    try:
        v = float(b**n)
    except (OverflowError, ZeroDivisionError):
        return None
    return v if math.isfinite(v) else None


def _simp(e):
    t = e[0]
    if t in ("num", "sym"):
        return e
    if t == "fn":
        a = _simp(e[2])
        name = e[1]
        if a[0] == "num":
            v = a[1]
            if v == 0 and name in ("sin", "tan", "asin", "atan", "sinh"):
                return _num(0)
            if v == 0 and name in ("cos", "exp", "cosh"):
                return _num(1)
            if v == 1 and name == "log":
                return _num(0)
        if name == "log" and a[0] == "fn" and a[1] == "exp":
            return a[2]
        if name == "exp" and a[0] == "fn" and a[1] == "log":
            return a[2]
        return ("fn", name, a)
    if t == "pow":
        b = _simp(e[1])
        x = _simp(e[2])
        if x[0] == "num":
            if x[1] == 0:
                return _num(1)
            if x[1] == 1:
                return b
            if b[0] == "num" and _isint(x[1]) and b[1] != 0 and _npow(b[1], x[1]) is not None:
                return _num(_npow(b[1], x[1]))
            if b[0] == "num" and b[1] == 1:
                return _num(1)
            if b[0] == "pow" and b[2][0] == "num" and _isint(x[1]):
                return _simp(("pow", b[1], _num(b[2][1] * x[1])))
            if b[0] == "mul" and _isint(x[1]):
                return _simp(("mul", [("pow", f, x) for f in b[1]]))
        return ("pow", b, x)
    if t == "mul":
        fs = []
        for f in e[1]:
            f = _simp(f)
            if f[0] == "mul":
                fs.extend(f[1])
            else:
                fs.append(f)
        coef = 1.0
        bases = {}
        order = []
        for f in fs:
            if f[0] == "num":
                coef *= f[1]
                continue
            if f[0] == "pow" and f[2][0] == "num":
                b, xv = f[1], f[2][1]
            else:
                b, xv = f, 1.0
            k = _key(b)
            if k not in bases:
                bases[k] = [b, 0.0]
                order.append(k)
            bases[k][1] += xv
        if coef == 0:
            return _num(0)
        out = []
        for k in sorted(order):
            b, xv = bases[k]
            if xv == 0:
                continue
            p = b if xv == 1 else _simp_pow(b, xv)
            if p[0] == "num":
                coef *= p[1]
            else:
                out.append(p)
        if not out:
            return _num(coef)
        if coef == 1 and len(out) == 1:
            return out[0]
        return ("mul", ([_num(coef)] if coef != 1 else []) + out)
    # add
    ts = []
    for f in e[1]:
        f = _simp(f)
        if f[0] == "add":
            ts.extend(f[1])
        else:
            ts.append(f)
    const = 0.0
    terms = {}
    order = []
    for f in ts:
        if f[0] == "num":
            const += f[1]
            continue
        c, rest = _split_coef(f)
        k = _key(rest)
        if k not in terms:
            terms[k] = [rest, 0.0]
            order.append(k)
        terms[k][1] += c
    out = []
    for k in sorted(order):
        rest, c = terms[k]
        if c == 0:
            continue
        out.append(rest if c == 1 else _simp(("mul", [_num(c), rest])))
    if const != 0:
        out.append(_num(const))
    if not out:
        return _num(0)
    if len(out) == 1:
        return out[0]
    return ("add", out)


def _simp_pow(b, xv):
    if b[0] == "num" and _isint(xv) and b[1] != 0 and _npow(b[1], xv) is not None:
        return _num(_npow(b[1], xv))
    return ("pow", b, _num(xv))


def _split_coef(f):
    if f[0] == "mul" and f[1][0][0] == "num":
        rest = f[1][1:]
        return f[1][0][1], (rest[0] if len(rest) == 1 else ("mul", rest))
    return 1.0, f


def _add(*a):
    return _simp(("add", list(a)))


def _mul(*a):
    return _simp(("mul", list(a)))


def _pow(b, x):
    return _simp(("pow", b, x if isinstance(x, tuple) else _num(x)))


def _fn(name, a):
    return _simp(("fn", name, a))


# ---------------------------------------------------------------- parsing
def _parse(expr):
    rpn = shunting_yard(str(expr)).rpn
    st = []
    for tok in rpn:
        try:
            st.append(_num(float(tok)))
            continue
        except ValueError:
            pass
        if tok == "neg":
            st.append(("mul", [_num(-1), st.pop()]))
        elif tok in ("+", "-", "*", "/", "^"):
            b = st.pop()
            a = st.pop()
            if tok == "+":
                st.append(("add", [a, b]))
            elif tok == "-":
                st.append(("add", [a, ("mul", [_num(-1), b])]))
            elif tok == "*":
                st.append(("mul", [a, b]))
            elif tok == "/":
                st.append(("mul", [a, ("pow", b, _num(-1))]))
            else:
                st.append(("pow", a, b))
        elif tok == "sqrt":
            st.append(("pow", st.pop(), _num(0.5)))
        elif tok in _FNS:
            st.append(("fn", tok, st.pop()))
        elif tok in ("abs", "min", "max"):
            raise ValueError(f"{tok} is not supported")
        elif tok == "pi":
            st.append(_num(math.pi))
        else:
            st.append(("sym", tok))
    if len(st) != 1:
        raise ValueError("malformed expression")
    return _simp(st[0])


# ---------------------------------------------------------------- differentiation
def _d(e, x):
    t = e[0]
    if _free(e, x):
        return _num(0)
    if t == "sym":
        return _num(1)
    if t == "add":
        return _simp(("add", [_d(f, x) for f in e[1]]))
    if t == "mul":
        fs = e[1]
        terms = []
        for i in range(len(fs)):
            terms.append(("mul", [_d(fs[i], x)] + [fs[j] for j in range(len(fs)) if j != i]))
        return _simp(("add", terms))
    if t == "pow":
        b, n = e[1], e[2]
        if _free(n, x):
            return _mul(n, _pow(b, _add(n, _num(-1))), _d(b, x))
        if _free(b, x):
            return _mul(e, _fn("log", b), _d(n, x))
        return _mul(e, _add(_mul(_d(n, x), _fn("log", b)), _mul(n, _d(b, x), _pow(b, -1))))
    u = e[2]
    du = _d(u, x)
    name = e[1]
    if name == "sin":
        g = _fn("cos", u)
    elif name == "cos":
        g = _mul(_num(-1), _fn("sin", u))
    elif name == "tan":
        g = _pow(_fn("cos", u), -2)
    elif name == "exp":
        g = e
    elif name == "log":
        g = _pow(u, -1)
    elif name == "asin":
        g = _pow(_add(_num(1), _mul(_num(-1), _pow(u, 2))), -0.5)
    elif name == "acos":
        g = _mul(_num(-1), _pow(_add(_num(1), _mul(_num(-1), _pow(u, 2))), -0.5))
    elif name == "atan":
        g = _pow(_add(_num(1), _pow(u, 2)), -1)
    elif name == "sinh":
        g = _fn("cosh", u)
    else:
        g = _fn("sinh", u)
    return _mul(g, du)


# ---------------------------------------------------------------- numeric evaluation
def _ev(e, env):
    t = e[0]
    if t == "num":
        return e[1]
    if t == "sym":
        return env[e[1]]
    if t == "add":
        s = 0.0
        for f in e[1]:
            s += _ev(f, env)
        return s
    if t == "mul":
        s = 1.0
        for f in e[1]:
            s *= _ev(f, env)
        return s
    if t == "pow":
        b, n = _ev(e[1], env), _ev(e[2], env)
        if b == 0 and n < 0:
            return math.nan
        if b < 0 and not (math.isfinite(n) and n == int(n)):
            return math.nan
        try:
            return float(b**n)
        except OverflowError:
            return math.inf
    a = _ev(e[2], env)
    try:
        if e[1] == "log":
            return math.log(a) if a > 0 else math.nan
        if e[1] in ("asin", "acos") and abs(a) > 1:
            return math.nan
        return float(getattr(math, e[1])(a))
    except (ValueError, OverflowError):
        return math.nan


# ---------------------------------------------------------------- polynomials and rational functions
def _padd(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0.0) + (q[i] if i < len(q) else 0.0) for i in range(n)]


def _pmul(p, q):
    out = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _ptrim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _poly(e, x):
    """Ascending coefficients of a polynomial in x, or None."""
    t = e[0]
    if _free(e, x):
        return [_ev_const(e)] if _ev_const(e) is not None else None
    if t == "sym":
        return [0.0, 1.0]
    if t == "add":
        out = [0.0]
        for f in e[1]:
            p = _poly(f, x)
            if p is None:
                return None
            out = _padd(out, p)
        return _ptrim(out)
    if t == "mul":
        out = [1.0]
        for f in e[1]:
            p = _poly(f, x)
            if p is None:
                return None
            out = _pmul(out, p)
        return _ptrim(out)
    if t == "pow" and e[2][0] == "num" and _isint(e[2][1]) and 0 < e[2][1] <= 32:
        p = _poly(e[1], x)
        if p is None:
            return None
        out = [1.0]
        for _ in range(int(e[2][1])):
            out = _pmul(out, p)
        return _ptrim(out)
    return None


def _ev_const(e):
    try:
        v = _ev(e, {})
    except KeyError:
        return None
    return v if math.isfinite(v) else None


def _rat(e, x):
    """(numerator, denominator) ascending coefficients of a rational function in x, or None."""
    p = _poly(e, x)
    if p is not None:
        return p, [1.0]
    t = e[0]
    if t == "add":
        N, D = [0.0], [1.0]
        for f in e[1]:
            r = _rat(f, x)
            if r is None:
                return None
            N, D = _padd(_pmul(N, r[1]), _pmul(r[0], D)), _pmul(D, r[1])
        return _ptrim(N), _ptrim(D)
    if t == "mul":
        N, D = [1.0], [1.0]
        for f in e[1]:
            r = _rat(f, x)
            if r is None:
                return None
            N, D = _pmul(N, r[0]), _pmul(D, r[1])
        return _ptrim(N), _ptrim(D)
    if t == "pow" and e[2][0] == "num" and _isint(e[2][1]) and -32 <= e[2][1] < 0:
        p = _poly(e[1], x)
        if p is None:
            return None
        out = [1.0]
        for _ in range(int(-e[2][1])):
            out = _pmul(out, p)
        return [1.0], _ptrim(out)
    return None


def _pdivmod(N, D):
    N = list(N)
    q = [0.0] * max(len(N) - len(D) + 1, 1)
    dl = D[-1]
    for i in range(len(N) - len(D), -1, -1):
        c = N[i + len(D) - 1] / dl
        q[i] = c
        for j in range(len(D)):
            N[i + j] -= c * D[j]
    r = _ptrim(N[: len(D) - 1] or [0.0])
    return _ptrim(q), r


def _cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _cdiv(a, b):
    den = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den)


def _cpoly(p, z):
    """Horner with complex z on real ascending coefficients."""
    v = (0.0, 0.0)
    for c in reversed(p):
        v = _cmul(v, z)
        v = (v[0] + c, v[1])
    return v


def _roots(p):
    """Durand-Kerner roots of a real polynomial (ascending coefficients), polished by Newton."""
    n = len(p) - 1
    lead = p[-1]
    q = [c / lead for c in p]
    z = [(1.0, 0.0)]
    w = (0.4, 0.9)
    for _ in range(n - 1):
        z.append(_cmul(z[-1], w))
    z = [_cmul(zz, w) for zz in z]
    for _ in range(500):
        mx = 0.0
        new = []
        for i in range(n):
            den = (1.0, 0.0)
            for j in range(n):
                if j != i:
                    den = _cmul(den, (z[i][0] - z[j][0], z[i][1] - z[j][1]))
            step = _cdiv(_cpoly(q, z[i]), den)
            new.append((z[i][0] - step[0], z[i][1] - step[1]))
            mx = max(mx, abs(step[0]) + abs(step[1]))
        z = new
        if mx < 1e-15:
            break
    return z


def _snap(v):
    for den in range(1, 1001):
        r = round(v * den)
        if abs(v * den - r) < 1e-9 * max(1.0, abs(v * den)):
            return r / den
    return v


def _cluster(z, p):
    """Group roots within 1e-6 into (mean root, multiplicity), real when the imaginary part is negligible."""
    used = [False] * len(z)
    out = []
    for i in range(len(z)):
        if used[i]:
            continue
        grp = [j for j in range(len(z)) if not used[j] and abs(z[j][0] - z[i][0]) + abs(z[j][1] - z[i][1]) < 1e-6]
        for j in grp:
            used[j] = True
        re = 0.0
        im = 0.0
        for j in grp:
            re += z[j][0]
            im += z[j][1]
        re, im = re / len(grp), im / len(grp)
        if len(grp) > 1:
            # a root of multiplicity mu is a simple root of the (mu - 1)-th derivative: Newton there
            dp = list(p)
            for _ in range(len(grp) - 1):
                dp = [k * dp[k] for k in range(1, len(dp))]
            d2 = [k * dp[k] for k in range(1, len(dp))]
            for _ in range(30):
                den = _cpoly(d2, (re, im))
                if den[0] == 0 and den[1] == 0:
                    break
                st = _cdiv(_cpoly(dp, (re, im)), den)
                re, im = re - st[0], im - st[1]
        if abs(im) < 1e-9 * max(1.0, abs(re)):
            im = 0.0
        out.append(((_snap(re), _snap(im)), len(grp)))
    return out


def _taylor_shift(p, r):
    """Coefficients of p(t + r) for complex r (ascending)."""
    return _taylor_cshift([(c, 0.0) for c in p], r)


def _series_div(a, b, m):
    """First m coefficients of the power series a(t) / b(t) (complex coefficient lists)."""
    out = []
    for k in range(m):
        s = a[k] if k < len(a) else (0.0, 0.0)
        for j in range(1, k + 1):
            if j < len(b):
                pr = _cmul(b[j], out[k - j])
                s = (s[0] - pr[0], s[1] - pr[1])
        out.append(_cdiv(s, b[0]))
    return out


def _int_rational(N, D, x):
    X = ("sym", x)
    q, r = _pdivmod(N, D)
    terms = []
    for k, c in enumerate(q):
        if c != 0:
            terms.append(_mul(_num(_snap(c / (k + 1))), _pow(X, k + 1)))
    if len(r) == 1 and r[0] == 0:
        return _simp(("add", terms)) if terms else _num(0)
    if len(D) - 1 > 12:
        return None
    roots = _cluster(_roots(D), D)
    if sum(mu for _, mu in roots) != len(D) - 1:
        return None
    for (zr, zi), mu in roots:
        if zi < 0:
            continue
        # D(x) = (x - z)^mu * D1(x)
        D1 = [(c, 0.0) for c in D]
        for _ in range(mu):
            # synthetic division by (x - z)
            n = len(D1) - 1
            out = [(0.0, 0.0)] * n
            acc = (0.0, 0.0)
            for i in range(n, 0, -1):
                acc = (D1[i][0] + _cmul(acc, (zr, zi))[0], D1[i][1] + _cmul(acc, (zr, zi))[1])
                out[i - 1] = acc
            D1 = out
        a = _taylor_shift(r, (zr, zi))
        bshift = _taylor_cshift(D1, (zr, zi))
        g = _series_div(a, bshift, mu)
        # coefficient of 1/(x - z)^j is g[mu - j]
        for j in range(1, mu + 1):
            c = g[mu - j]
            c = (_snap(c[0]), _snap(c[1]))
            if zi == 0:
                if c[0] == 0:
                    continue
                lin = _add(X, _num(-zr))
                if j == 1:
                    terms.append(_mul(_num(c[0]), _fn("log", lin)))
                else:
                    terms.append(_mul(_num(_snap(-c[0] / (j - 1))), _pow(lin, -(j - 1))))
            else:
                if j > 1:
                    return None
                quad = _add(_pow(_add(X, _num(-zr)), 2), _num(_snap(zi * zi)))
                if c[0] != 0:
                    terms.append(_mul(_num(c[0]), _fn("log", quad)))
                if c[1] != 0:
                    terms.append(
                        _mul(_num(_snap(-2.0 * c[1])), _fn("atan", _mul(_add(X, _num(-zr)), _num(_snap(1.0 / zi)))))
                    )
    return _simp(("add", terms)) if terms else _num(0)


def _taylor_cshift(p, r):
    out = list(p)
    n = len(out)
    for k in range(n - 1):
        for i in range(n - 2, k - 1, -1):
            m = _cmul(out[i + 1], r)
            out[i] = (out[i][0] + m[0], out[i][1] + m[1])
    return out


# ---------------------------------------------------------------- integration
def _linear(u, x):
    p = _poly(u, x)
    if p is not None and len(p) == 2 and p[1] != 0:
        return p[1], p[0]
    return None


def _table(e, x):
    X = ("sym", x)
    t = e[0]
    if t == "sym":
        return _mul(_num(0.5), _pow(X, 2))
    if t == "pow":
        b, n = e[1], e[2]
        if n == ("num", -0.5):
            q = _poly(b, x)
            if q is not None and len(q) == 3 and q[1] == 0 and q[0] != 0:
                c, a2 = q[0], q[2]
                if a2 < 0 < c:
                    return _mul(_num(_snap(1 / math.sqrt(-a2))), _fn("asin", _mul(X, _num(_snap(math.sqrt(-a2 / c))))))
                if a2 > 0:
                    return _mul(
                        _num(_snap(1 / math.sqrt(a2))),
                        _fn("log", _add(_mul(_num(_snap(math.sqrt(a2))), X), _pow(b, 0.5))),
                    )
        lin = _linear(b, x)
        if lin and _free(n, x):
            a = lin[0]
            if n == ("num", -1.0):
                return _mul(_num(_snap(1 / a)), _fn("log", b))
            n1 = _add(n, _num(1))
            return _mul(_pow(b, n1), _pow(_mul(_num(a), n1), -1))
        lin = _linear(n, x)
        if lin and _free(b, x):
            return _mul(e, _pow(_mul(_num(lin[0]), _fn("log", b)), -1))
        return None
    if t == "fn":
        lin = _linear(e[2], x)
        if not lin:
            return None
        u = e[2]
        inv = _num(_snap(1 / lin[0]))
        name = e[1]
        if name == "sin":
            return _mul(_num(-1), inv, _fn("cos", u))
        if name == "cos":
            return _mul(inv, _fn("sin", u))
        if name == "tan":
            return _mul(_num(-1), inv, _fn("log", _fn("cos", u)))
        if name == "exp":
            return _mul(inv, e)
        if name == "log":
            return _mul(inv, _add(_mul(u, e), _mul(_num(-1), u)))
        if name == "sinh":
            return _mul(inv, _fn("cosh", u))
        if name == "cosh":
            return _mul(inv, _fn("sinh", u))
        root = _pow(_add(_num(1), _mul(_num(-1), _pow(u, 2))), 0.5)
        if name == "asin":
            return _mul(inv, _add(_mul(u, e), root))
        if name == "acos":
            return _mul(inv, _add(_mul(u, e), _mul(_num(-1), root)))
        if name == "atan":
            return _mul(inv, _add(_mul(u, e), _mul(_num(-0.5), _fn("log", _add(_num(1), _pow(u, 2))))))
    return None


def _factors(e):
    return list(e[1]) if e[0] == "mul" else [e]


def _parts(fs, x, depth):
    """Polynomial times exp/sin/cos/sinh/cosh (tabular parts) or times log/atan/asin/acos (one step)."""
    X = ("sym", x)
    polys = [f for f in fs if _poly(f, x) is not None]
    rest = [f for f in fs if _poly(f, x) is None]
    if len(rest) != 1 or not polys:
        return None
    g = rest[0]
    P = _simp(("mul", polys))
    if g[0] == "fn" and g[1] in ("exp", "sin", "cos", "sinh", "cosh") and _linear(g[2], x):
        out = []
        sign = 1.0
        cur = g
        Pk = P
        for _ in range(40):
            cur = _integrate(cur, x, depth + 1)
            if cur is None:
                return None
            out.append(_mul(_num(sign), Pk, cur))
            Pk = _d(Pk, x)
            sign = -sign
            if Pk == ("num", 0.0):
                return _simp(("add", out))
        return None
    if g[0] == "fn" and g[1] in ("log", "atan", "asin", "acos") and _linear(g[2], x):
        Q = _integrate(P, x, depth + 1)
        if Q is None:
            return None
        rest_int = _integrate(_mul(Q, _d(g, x)), x, depth + 1)
        if rest_int is None:
            return None
        return _add(_mul(Q, g), _mul(_num(-1), rest_int))
    return None


def _exp_trig(fs, x):
    if len(fs) != 2:
        return None
    ex = [f for f in fs if f[0] == "fn" and f[1] == "exp"]
    tr = [f for f in fs if f[0] == "fn" and f[1] in ("sin", "cos")]
    if len(ex) != 1 or len(tr) != 1:
        return None
    la, lb = _linear(ex[0][2], x), _linear(tr[0][2], x)
    if not la or not lb:
        return None
    a, b = la[0], lb[0]
    v = tr[0][2]
    s, c = _fn("sin", v), _fn("cos", v)
    k = _num(_snap(1 / (a * a + b * b)))
    if tr[0][1] == "sin":
        inner = _add(_mul(_num(a), s), _mul(_num(-b), c))
    else:
        inner = _add(_mul(_num(a), c), _mul(_num(b), s))
    return _mul(k, ex[0], inner)


def _subexprs(e, out):
    if e[0] in ("fn", "pow"):
        out.append(e)
    if e[0] == "fn":
        _subexprs(e[2], out)
    elif e[0] == "pow":
        _subexprs(e[1], out)
        _subexprs(e[2], out)
    elif e[0] in ("add", "mul"):
        for f in e[1]:
            _subexprs(f, out)
    return out


def _subst(e, old, new):
    if e == old:
        return new
    t = e[0]
    if t == "fn":
        return ("fn", e[1], _subst(e[2], old, new))
    if t == "pow":
        return ("pow", _subst(e[1], old, new), _subst(e[2], old, new))
    if t in ("add", "mul"):
        return (t, [_subst(f, old, new) for f in e[1]])
    return e


def _usub(e, x, depth):
    seen = set()
    for g in _subexprs(e, []):
        k = _key(g)
        if k in seen or _free(g, x) or _linear(g, x) is not None:
            continue
        seen.add(k)
        for cand in (g,) if g[0] == "fn" else (g, g[1]):
            if _free(cand, x) or _linear(cand, x) is not None or cand == ("sym", x):
                continue
            dg = _d(cand, x)
            if dg == ("num", 0.0):
                continue
            u = ("sym", "_u")
            h = _simp(("mul", [_subst(e, cand, u), ("pow", dg, _num(-1))]))
            if not _free(h, x):
                continue
            H = _integrate(h, "_u", depth + 1)
            if H is not None:
                return _simp(_subst(H, u, cand))
    return None


def _expand(e):
    """Distribute products over sums (one level)."""
    if e[0] != "mul":
        return None
    fs = e[1]
    for i, f in enumerate(fs):
        if f[0] == "add":
            others = fs[:i] + fs[i + 1 :]
            return _simp(("add", [("mul", others + [t]) for t in f[1]]))
    return None


def _integrate(e, x, depth=0):
    if depth > 8:
        return None
    X = ("sym", x)
    e = _simp(e)
    if _free(e, x):
        return _mul(e, X)
    if e[0] == "add":
        out = []
        for f in e[1]:
            r = _integrate(f, x, depth + 1)
            if r is None:
                return None
            out.append(r)
        return _simp(("add", out))
    fs = _factors(e)
    const = [f for f in fs if _free(f, x)]
    var = [f for f in fs if not _free(f, x)]
    if const:
        r = _integrate(_simp(("mul", var)), x, depth + 1)
        return None if r is None else _mul(*(const + [r]))
    r = _table(e, x)
    if r is not None:
        return r
    rt = _rat(e, x)
    if rt is not None:
        r = _int_rational(rt[0], rt[1], x)
        if r is not None:
            return r
    for attempt in (_parts, _exp_trig):
        r = attempt(fs, x, depth) if attempt is _parts else attempt(fs, x)
        if r is not None:
            return r
    r = _usub(e, x, depth)
    if r is not None:
        return r
    ex = _expand(e)
    if ex is not None and ex != e:
        return _integrate(ex, x, depth + 1)
    return None


def _check(F, f, x):
    dF = _d(F, x)
    worst = 0.0
    n = 0
    for v in (0.37, 1.13, 2.71, -0.83, 0.61):
        a, b = _ev(dF, {x: v}), _ev(f, {x: v})
        if math.isfinite(a) and math.isfinite(b):
            worst = max(worst, abs(a - b) / max(1.0, abs(b)))
            n += 1
    return (worst if n else math.nan), n


def symbolic_diff(expr, x: str = "x") -> RichResult:
    r"""Symbolic derivative of an elementary expression (sum, product, power and chain rules), simplified.

    Examples
    --------
    >>> symbolic_diff("x^3*sin(x)").derivative
    'cos(x)*x^3 + 3*sin(x)*x^2'
    """
    e = _parse(expr)
    d = _d(e, str(x))
    return RichResult(payload={"derivative": _str(d), "expression": _str(e)})


def symbolic_integrate(expr, x: str = "x") -> RichResult:
    r"""Antiderivative of an elementary expression by table, linearity, rational functions, parts and substitution.

    The expression (``+ - * / ^``, ``sqrt``, ``sin cos tan exp log asin
    acos atan sinh cosh``) is parsed with the shunting-yard parser
    (:func:`morie.fn.symalg.shunting_yard`) and brought to a canonical
    sum-of-products form (collected coefficients and powers, sorted
    terms). The integrator tries, in order: constants and linearity;
    the elementary table for arguments linear in ``x``; rational functions
    by polynomial division and partial fractions over the roots of the
    denominator (Durand-Kerner, repeated real roots by Taylor-series
    division, complex pairs giving ``log`` and ``atan`` terms) -- the
    rational-function part of the Risch-Bronstein algorithm; integration by
    parts for polynomials times ``exp sin cos sinh cosh`` (tabular) or
    ``log atan asin acos``; ``e^(ax) sin(bx)``-type products; the
    derivative-divides substitution ``u = g(x)`` over the functions and
    powers of the integrand; and distribution over sums. The result is
    checked by differentiating it back at five points (``max_rel_error``).
    Returns ``antiderivative`` (``None`` when no rule applies) and
    ``verified``.

    References
    ----------
    Bronstein, M. (1997). *Symbolic Integration I: Transcendental
    Functions*. Springer.
    Geddes, K. O., Czapor, S. R. and Labahn, G. (1992). *Algorithms for
    Computer Algebra*, chapters 11-12. Kluwer.
    Moses, J. (1971). Symbolic integration: the stormy decade.
    *Communications of the ACM*, 14, 548-560.

    Examples
    --------
    >>> symbolic_integrate("x*exp(2*x)").antiderivative
    '-0.25*exp(2*x) + 0.5*exp(2*x)*x'
    >>> symbolic_integrate("1/(x^2 + 1)").antiderivative
    'atan(x)'
    >>> symbolic_integrate("2*x*cos(x^2)").antiderivative
    'sin(x^2)'
    """
    x = str(x)
    e = _parse(expr)
    F = _integrate(e, x)
    if F is None:
        return RichResult(
            payload={"antiderivative": None, "verified": False, "max_rel_error": math.nan, "expression": _str(e)}
        )
    err, n = _check(F, e, x)
    return RichResult(
        payload={
            "antiderivative": _str(F),
            "verified": bool(n > 0 and err < 1e-8),
            "max_rel_error": err,
            "expression": _str(e),
        }
    )


def cheatsheet() -> str:
    return "symbolic_integrate('x*exp(2*x)') -> antiderivative string; symbolic_diff(expr) -> derivative."
