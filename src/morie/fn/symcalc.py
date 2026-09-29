# morie.fn -- function file (rootcoder007/morie)
"""Symbolic calculus: limits by series, L'Hopital's rule and exp-log order comparison, determinants, inverses,
characteristic polynomials and eigenvalues of matrices with symbolic entries, classification and closed-form
solution of first-order ODEs, and the rational case of the Risch algorithm (Hermite reduction plus
Rothstein-Trager). The expression core, the shunting-yard parser and the integrator are copied from
morie.fn.symalg and morie.fn.symint."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["symbolic_limit", "matrix_symbolic", "ode_symbolic", "risch_integration"]

_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4, "neg": 3}
_RIGHT = {"^", "neg"}
_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "abs": abs,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh,
}  # fmt: skip


def _tokenize(s):
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            if j < len(s) and s[j] in "eE" and j + 1 < len(s) and (s[j + 1].isdigit() or s[j + 1] in "+-"):
                j += 2
                while j < len(s) and s[j].isdigit():
                    j += 1
            out.append(s[i:j])
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            out.append(s[i:j])
            i = j
        elif ch in "+-*/^(),":
            out.append(ch)
            i += 1
        else:
            raise ValueError(f"unexpected character {ch!r}")
    return out


def _isnum(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def _call(f, x):
    try:
        return float(_FUNCS[f](x))
    except (ValueError, OverflowError):
        return math.inf if f == "exp" else math.nan


def _binop(t, a, b):
    if t == "+":
        return a + b
    if t == "-":
        return a - b
    if t == "*":
        return a * b
    if t == "/":
        if b == 0:
            return math.nan if a == 0 or math.isnan(a) else math.copysign(math.inf, a)
        return a / b
    if t == "min":
        return min(a, b)
    if t == "max":
        return max(a, b)
    if a == 0 and b < 0:
        return math.inf
    if a < 0 and b != int(b):
        return math.nan
    try:
        return float(a**b)
    except OverflowError:
        return math.inf


def _shunting_yard(tokens, variables=None):
    """Shunting-yard conversion of an infix expression to reverse Polish notation (copied from symalg)."""
    toks = _tokenize(tokens) if isinstance(tokens, str) else [str(t) for t in tokens]
    variables = dict(variables or {})
    out, st = [], []
    prev = None
    for t in toks:
        if _isnum(t):
            out.append(t)
        elif t in _FUNCS or t in ("min", "max"):
            st.append(t)
        elif t == ",":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("misplaced comma")
        elif t in _PREC:
            op = t
            if t == "-" and (prev is None or prev in _PREC or prev in ("(", ",")):
                op = "neg"
            elif t == "+" and (prev is None or prev in _PREC or prev in ("(", ",")):
                prev = t
                continue
            while st and st[-1] in _PREC and op != "neg":
                top = st[-1]
                if _PREC[top] > _PREC[op] or (_PREC[top] == _PREC[op] and op not in _RIGHT):
                    out.append(st.pop())
                else:
                    break
            st.append(op)
        elif t == "(":
            st.append(t)
        elif t == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("mismatched parentheses")
            st.pop()
            if st and (st[-1] in _FUNCS or st[-1] in ("min", "max")):
                out.append(st.pop())
        else:
            out.append(t)
        prev = t
    while st:
        if st[-1] == "(":
            raise ValueError("mismatched parentheses")
        out.append(st.pop())
    val = None
    if all(_isnum(t) or t in _PREC or t in _FUNCS or t in ("min", "max") or t in variables for t in out):
        vs = []
        for t in out:
            if _isnum(t):
                vs.append(float(t))
            elif t in variables:
                vs.append(float(variables[t]))
            elif t == "neg":
                vs.append(-vs.pop())
            elif t in _FUNCS:
                vs.append(_call(t, vs.pop()))
            else:
                b, a = vs.pop(), vs.pop()
                vs.append(_binop(t, a, b))
        if len(vs) != 1:
            raise ValueError("malformed expression")
        val = vs[0]
    return {"rpn": out, "value": val, "tokens": toks}


_FNS = ("sin", "cos", "tan", "exp", "log", "asin", "acos", "atan", "sinh", "cosh")


# ---------------------------------------------------------------- construction and printing
def _num(v):
    return ("num", float(v))


def _fmt(v):
    if v == 0:
        return "0"
    if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
        return f"{int(v):d}"
    return f"{v:.15g}"


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
    rpn = _shunting_yard(str(expr))["rpn"]
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
    inner = _add(_mul(_num(a), s), _mul(_num(-b), c)) if tr[0][1] == "sin" else _add(_mul(_num(a), c), _mul(_num(b), s))
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


# ---------------------------------------------------------------- polynomial tools for the Risch reduction
def _pdeg(p):
    p = _ptrim(p)
    return len(p) - 1 if not (len(p) == 1 and p[0] == 0) else -1


def _pmonic(p):
    return [c / p[-1] for c in p]


def _pder(p):
    return _ptrim([p[i] * i for i in range(1, len(p))]) if len(p) > 1 else [0.0]


def _pzero(p, tol=1e-10):
    m = max(abs(c) for c in p)
    return m <= tol


def _pclean(p, tol=1e-10):
    m = max(abs(c) for c in p) or 1.0
    return _ptrim([0.0 if abs(c) < tol * m else c for c in p])


def _pgcd(a, b):
    """Monic gcd of two float polynomials (Euclid with a relative tolerance)."""
    a, b = _pclean(a), _pclean(b)
    while not _pzero(b):
        _q, r = _pdivmod(a, b)
        a, b = b, _pclean(r)
    return _pmonic(a) if not _pzero(a) else [1.0]


def _pxgcd(a, b):
    """Extended Euclid: (g, s, t) with s a + t b = g, g monic."""
    r0, r1 = _pclean(a), _pclean(b)
    s0, s1 = [1.0], [0.0]
    t0, t1 = [0.0], [1.0]
    while not _pzero(r1):
        q, r = _pdivmod(r0, r1)
        r0, r1 = r1, _pclean(r)
        s0, s1 = s1, _pclean(_padd(s0, [-c for c in _pmul(q, s1)]))
        t0, t1 = t1, _pclean(_padd(t0, [-c for c in _pmul(q, t1)]))
    lead = r0[-1]
    return _pmonic(r0), [c / lead for c in s0], [c / lead for c in t0]


def _psquarefree(p):
    """Yun's squarefree decomposition: list of (multiplicity, factor) with p = prod factor^multiplicity."""
    p = _pmonic(_pclean(p))
    out = []
    c = _pgcd(p, _pder(p))
    w, _r = _pdivmod(p, c)
    w = _pclean(w)
    i = 1
    while _pdeg(w) > 0:
        y = _pgcd(w, c)
        g, _r = _pdivmod(w, y)
        g = _pclean(g)
        if _pdeg(g) > 0:
            out.append((i, _pmonic(g)))
        w = y
        c, _r = _pdivmod(c, y)
        c = _pclean(c)
        i += 1
    return out


def _pmod(a, m):
    if _pdeg(a) < _pdeg(m):
        return _pclean(a)
    _q, r = _pdivmod(a, m)
    return _pclean(r)


def _ppow(p, k):
    out = [1.0]
    for _ in range(k):
        out = _pmul(out, p)
    return out


def _poly_expr(p, x):
    X = ("sym", x)
    terms = []
    for k, c in enumerate(p):
        if c == 0:
            continue
        terms.append(_mul(_num(_snap(c)), _pow(X, k)) if k else _num(_snap(c)))
    return _simp(("add", terms)) if terms else _num(0)


def _hermite(A, D, x):
    """Hermite reduction: A/D = d/dx(g) + q + B/Dstar with Dstar squarefree. Returns (g, q, B, Dstar)."""
    A, D = _pclean(A), _pmonic(_pclean(D))
    parts = []  # (numerator, denominator) of the accumulated rational part
    while True:
        sf = _psquarefree(D)
        top = max((j for j, f in sf if j >= 2 and _pdeg(f) > 0), default=0)
        if top == 0:
            break
        V = next(f for j, f in sf if j == top)
        U, _r = _pdivmod(D, _ppow(V, top))
        U = _pclean(U)
        # A/(U V^top) = A1/V^top + A2/U with A = A1 U + A2 V^top (Bezout, gcd(U, V^top) = 1)
        g, s, t = _pxgcd(U, _ppow(V, top))
        if _pdeg(g) > 0:
            return None
        A1 = _pmod(_pmul(A, s), _ppow(V, top))
        A2, _r = _pdivmod(_pclean(_padd(A, [-c for c in _pmul(A1, U)])), _ppow(V, top))
        A2 = _pclean(A2)
        # A1/V^top: s1 V + t1 V' = 1, integrate A1 t1 V'/V^top by parts
        _g1, s1, t1 = _pxgcd(V, _pder(V))
        s1 = _pmul(A1, s1)
        t1 = _pmul(A1, t1)
        j = top - 1
        parts.append(([-c / j for c in t1], _ppow(V, j)))
        rest = _padd(s1, [c / j for c in _pder(t1)])
        A = _pclean(_padd(_pmul(rest, U), _pmul(A2, _ppow(V, j))))
        D = _pmonic(_pmul(U, _ppow(V, j)))
    q, A = _pdivmod(A, D)
    g = _num(0)
    for num, den in parts:
        g = _add(g, _mul(_poly_expr(_pclean(num), x), _pow(_poly_expr(_pclean(den), x), -1)))
    return _simp(g), _pclean(q), _pclean(A), D


def _rothstein_trager(B, D, x):
    """Logarithmic part sum_i c_i log(x - a_i) of B/D with D squarefree, from the residues c_i = B(a_i)/D'(a_i)."""
    X = ("sym", x)
    if _pdeg(D) == 0:
        return _num(0)
    roots = _cluster(_roots(D), D)
    if sum(mu for _, mu in roots) != _pdeg(D) or any(mu != 1 for _, mu in roots):
        return None
    Dp = _pder(D)
    terms = []
    for (zr, zi), _mu in roots:
        if zi < 0:
            continue
        cb = _cpoly(B, (zr, zi))
        cd = _cpoly(Dp, (zr, zi))
        c = _cdiv(cb, cd)
        c = (_snap(c[0]), _snap(c[1]))
        if zi == 0:
            if c[0] != 0:
                terms.append(_mul(_num(c[0]), _fn("log", _add(X, _num(_snap(-zr))))))
        else:
            quad = _add(_pow(_add(X, _num(_snap(-zr))), 2), _num(_snap(zi * zi)))
            if c[0] != 0:
                terms.append(_mul(_num(c[0]), _fn("log", quad)))
            if c[1] != 0:
                terms.append(
                    _mul(_num(_snap(-2.0 * c[1])), _fn("atan", _mul(_add(X, _num(_snap(-zr))), _num(_snap(1.0 / zi)))))
                )
    return _simp(("add", terms)) if terms else _num(0)


def _generators(e, x, out):
    t = e[0]
    if t == "fn":
        out.add(e[1])
        _generators(e[2], x, out)
    elif t == "pow":
        if e[2][0] != "num" or not _isint(e[2][1]):
            out.add("algebraic" if not _free(e[1], x) else "constant power")
        _generators(e[1], x, out)
        _generators(e[2], x, out)
    elif t in ("add", "mul"):
        for f in e[1]:
            _generators(f, x, out)
    return out


def risch_integration(expr, x: str = "x") -> RichResult:
    r"""Risch-style integration of the rational part: Hermite reduction plus Rothstein-Trager, with the transcendental tower reported as out of scope.

    For a rational integrand ``f = N/D`` the three steps of the rational
    case of the Risch-Bronstein algorithm are carried out:

    1. Euclidean division ``N = q D + r`` gives the polynomial part
       ``int q dx``;
    2. Hermite reduction writes ``r/D = d/dx(g) + B/D*`` with ``D*``
       squarefree, using Yun's squarefree decomposition and Bezout
       (extended Euclid) identities, so the rational part ``g`` is found by
       algebra alone;
    3. the Rothstein-Trager residues ``c_i = B(a_i)/D*'(a_i)`` at the roots
       of ``D*`` give the logarithmic part ``sum_i c_i log(x - a_i)``,
       conjugate pairs combined into real ``log`` and ``atan`` terms.

    Out of scope, and reported as such rather than guessed: the
    transcendental tower (integrands containing ``exp``, ``log`` or the
    trigonometric functions, where Risch's structure theorem and the
    differential-algebra reduction over each new generator would be needed)
    and algebraic extensions (fractional powers of expressions in ``x``).
    :func:`morie.fn.symint.symbolic_integrate` covers many of those cases by
    other rules.

    Coefficients are double precision, so the roots of ``D*`` come from the
    Durand-Kerner iteration and residues are snapped to nearby rationals
    with a denominator of at most 1000.

    Parameters
    ----------
    expr : str
        Integrand.
    x : str
        Integration variable.

    Returns
    -------
    RichResult
        ``integral`` (string, ``None`` when out of scope), the
        ``polynomial_part``, ``rational_part`` and ``log_part``,
        ``rational`` (was the integrand a rational function),
        ``out_of_scope`` (reason or ``None``), ``generators`` (the
        transcendental or algebraic generators found), ``verified`` and
        ``max_rel_error`` from differentiating the answer back.

    References
    ----------
    Risch, R. H. (1969). The problem of integration in finite terms.
    *Transactions of the American Mathematical Society*, 139, 167-189.
    Bronstein, M. (1997). *Symbolic Integration I: Transcendental
    Functions*. Springer, sections 2.2 (Hermite reduction) and 2.4
    (Rothstein-Trager).
    Rothstein, M. (1976). *Aspects of symbolic integration and
    simplification of exponential and primitive functions*. PhD thesis,
    University of Wisconsin.
    Yun, D. Y. Y. (1976). On square-free decomposition algorithms.
    *Proceedings of SYMSAC 76*, 26-35.

    Examples
    --------
    >>> risch_integration("(x^2 + 1)/(x^3 - x)").integral
    'log(x + 1) + log(x - 1) - log(x)'
    >>> r = risch_integration("x/(x^2 + 2*x + 1)")
    >>> r.rational_part, r.log_part
    ('-x*(x + 1)^(-1)', 'log(x + 1)')
    >>> risch_integration("exp(x^2)").out_of_scope
    'the transcendental tower (generators: exp) is not implemented'
    """
    e = _parse(expr)
    gens = sorted(_generators(e, x, set()))
    rat = _rat(e, x)
    if rat is None:
        why = (
            "the transcendental tower (generators: "
            + ", ".join(g for g in gens if g not in ("algebraic",))
            + ") is not implemented"
            if any(g not in ("algebraic", "constant power") for g in gens)
            else "algebraic extensions (fractional powers of x) are not implemented"
        )
        return RichResult(
            payload={
                "integral": None,
                "polynomial_part": None,
                "rational_part": None,
                "log_part": None,
                "rational": False,
                "out_of_scope": why,
                "generators": gens,
                "verified": False,
                "max_rel_error": math.nan,
            }
        )
    N, D = rat
    lead = D[-1]
    N = [c / lead for c in N]
    D = _pmonic(D)
    q, r = _pdivmod(N, D)
    poly_part = _num(0)
    for k, c in enumerate(q):
        if c != 0:
            poly_part = _add(poly_part, _mul(_num(_snap(c / (k + 1))), _pow(("sym", x), k + 1)))
    hr = _hermite(_pclean(r), D, x) if _pdeg(D) > 0 else (_num(0), [0.0], [0.0], [1.0])
    if hr is None:
        return RichResult(
            payload={
                "integral": None,
                "polynomial_part": _str(poly_part),
                "rational_part": None,
                "log_part": None,
                "rational": True,
                "out_of_scope": "the Hermite reduction failed on this denominator",
                "generators": gens,
                "verified": False,
                "max_rel_error": math.nan,
            }
        )
    g, qextra, B, Dstar = hr
    for k, c in enumerate(qextra):
        if c != 0:
            poly_part = _add(poly_part, _mul(_num(_snap(c / (k + 1))), _pow(("sym", x), k + 1)))
    poly_part = _simp(poly_part)
    logs = _rothstein_trager(_pclean(B), Dstar, x) if not _pzero(B) else _num(0)
    if logs is None:
        return RichResult(
            payload={
                "integral": None,
                "polynomial_part": _str(poly_part),
                "rational_part": _str(g),
                "log_part": None,
                "rational": True,
                "out_of_scope": "the roots of the squarefree part were not separated",
                "generators": gens,
                "verified": False,
                "max_rel_error": math.nan,
            }
        )
    total = _simp(_add(_add(poly_part, g), logs))
    err, nck = _check(total, e, x)
    return RichResult(
        payload={
            "integral": _str(total),
            "polynomial_part": _str(poly_part),
            "rational_part": _str(g),
            "log_part": _str(logs),
            "rational": True,
            "out_of_scope": None,
            "generators": gens,
            "verified": bool(nck > 0 and err < 1e-8),
            "max_rel_error": err,
        }
    )


# ---------------------------------------------------------------- limits
_LIM_MAX = 12


def _num_at(e, x, v):
    try:
        return _ev(e, {x: v})
    except (KeyError, ZeroDivisionError, ValueError, OverflowError):
        return math.nan


def _continuous_at(e, x, x0, v, side):
    """Is the value at x0 the limit? A nearby value must agree with it, so that 1^inf and its like are rejected."""
    hs = []
    if side in ("both", "+"):
        hs.append(1e-6)
    if side in ("both", "-"):
        hs.append(-1e-6)
    for h in hs:
        w = _num_at(e, x, x0 + h)
        if not math.isfinite(w) or abs(w - v) > 1e-3 * max(1.0, abs(v)):
            return False
    return True


def _split_ratio(e, x):
    """(numerator, denominator) of an expression, splitting negative powers out of a product."""
    fs = _factors(e)
    nu, de = [], []
    for f in fs:
        if f[0] == "pow" and f[2][0] == "num" and f[2][1] < 0:
            de.append(_pow(f[1], -f[2][1]))
        else:
            nu.append(f)
    return (_simp(("mul", nu)) if nu else _num(1)), (_simp(("mul", de)) if de else _num(1))


def _series_coefs(e, x, x0, order):
    """Taylor coefficients f^(k)(x0)/k! for k = 0..order, or None when a derivative is not finite at x0."""
    out = []
    d = e
    fact = 1.0
    for k in range(order + 1):
        if k:
            d = _d(d, x)
            fact *= k
        v = _num_at(d, x, x0)
        if not math.isfinite(v):
            return None
        out.append(v / fact)
    return out


def _lead(cs, tol=1e-9):
    for k, c in enumerate(cs):
        if abs(c) > tol:
            return k, c
    return None


def _rat_limit_inf(e, x):
    """Exact limit at +infinity of a rational function, from the degrees and leading coefficients."""
    r = _rat(e, x)
    if r is None:
        return None
    N, D = _pclean(r[0]), _pclean(r[1])
    if _pzero(D):
        return None
    if _pzero(N):
        return 0.0
    dn, dd = _pdeg(N), _pdeg(D)
    if dn < dd:
        return 0.0
    if dn == dd:
        return _snap(N[-1] / D[-1])
    return math.inf if N[-1] / D[-1] > 0 else -math.inf


_BOUNDED = ("sin", "cos", "atan")


def _is_bounded(e, x):
    """Is the expression bounded on every neighbourhood of infinity (sines, cosines, arctangents, constants)?"""
    t = e[0]
    if t == "num":
        return True
    if t == "sym":
        return e[1] != x
    if t == "fn":
        return e[1] in _BOUNDED
    if t == "mul":
        return all(_is_bounded(f, x) for f in e[1])
    if t == "add":
        return all(_is_bounded(f, x) for f in e[1])
    if t == "pow":
        return _is_bounded(e[1], x) and e[2][0] == "num" and e[2][1] >= 0
    return False


def _asym(e, x):
    """Asymptotic order of an exp-log-power expression at +infinity.

    Returns a list of monomials ``(coefficient, {exponent: coefficient} of the
    polynomial inside exp, power of x, power of log x)``, or ``None`` when the
    expression is outside that class.
    """
    t = e[0]
    if t == "num":
        return [(e[1], {}, 0.0, 0.0)]
    if t == "sym":
        return [(1.0, {}, 1.0, 0.0)] if e[1] == x else None
    if t == "add":
        out = []
        for f in e[1]:
            m = _asym(f, x)
            if m is None:
                return None
            out += m
        return out
    if t == "mul":
        out = [(1.0, {}, 0.0, 0.0)]
        for f in e[1]:
            m = _asym(f, x)
            if m is None:
                return None
            out = [_amul(a, b) for a in out for b in m]
        return out
    if t == "pow":
        if e[2][0] != "num":
            return None
        m = _asym(e[1], x)
        if m is None or len(m) != 1:
            return None
        return [_apow(m[0], e[2][1])]
    if t == "fn" and e[1] == "exp":
        m = _asym(e[2], x)
        if m is None:
            return None
        p = {}
        for c, ep, a, b in m:
            if ep or b:
                return None
            p[a] = p.get(a, 0.0) + c
        return (
            [(1.0, {k: v for k, v in p.items() if k != 0.0 and v != 0.0}, 0.0, 0.0)]
            if 0.0 not in p
            else [(math.exp(p[0.0]), {k: v for k, v in p.items() if k != 0.0 and v != 0.0}, 0.0, 0.0)]
        )
    if t == "fn" and e[1] == "log":
        m = _asym(e[2], x)
        if m is None or len(m) != 1:
            return None
        c, ep, a, b = m[0]
        if ep or b or c <= 0:
            return None
        out = [(a, {}, 0.0, 1.0)]
        if c != 1.0:
            out.append((math.log(c), {}, 0.0, 0.0))
        return out
    return None


def _amul(a, b):
    p = dict(a[1])
    for k, v in b[1].items():
        p[k] = p.get(k, 0.0) + v
    return (a[0] * b[0], {k: v for k, v in p.items() if v != 0.0}, a[2] + b[2], a[3] + b[3])


def _apow(a, k):
    if a[0] < 0 and k != int(k):
        raise ValueError("a negative coefficient raised to a fractional power")
    return (a[0] ** k, {e: c * k for e, c in a[1].items()}, a[2] * k, a[3] * k)


def _aorder(m):
    """Comparison key of a monomial at +infinity: the exp part dominates, then the power of x, then of log x."""
    if m[1]:
        d = max(m[1])
        return (1, d, m[1][d], m[2], m[3]) if m[1][d] > 0 else (-1, -d, -m[1][d], -m[2], -m[3])
    return (0, 0.0, 0.0, m[2], m[3])


def _asym_limit(e, x):
    """Limit at +infinity from the asymptotic orders, or None when the class does not apply."""
    ms = _asym(e, x)
    if ms is None:
        return None
    ms = [m for m in ms if m[0] != 0.0]
    if not ms:
        return 0.0
    best = max(_aorder(m) for m in ms)
    tops = [m for m in ms if _aorder(m) == best]
    c = math.fsum(m[0] for m in tops)
    if best[0] == 0 and best[3] == 0.0 and best[4] == 0.0:
        rest = [m for m in ms if _aorder(m) != best]
        lower = all(_aorder(m) < best for m in rest)
        return c if lower else None
    if best > (0, 0.0, 0.0, 0.0, 0.0):
        if c == 0.0:
            return None
        return math.inf if c > 0 else -math.inf
    return 0.0


def symbolic_limit(expr, x: str = "x", x0=0.0, side: str = "both") -> RichResult:
    r"""Limit of an elementary expression by substitution, Taylor series, L'Hopital's rule and exp-log order comparison.

    The rules are tried in order:

    1. **substitution** -- when the expression evaluates to a finite value
       at ``x0`` it is continuous there and that value is the limit;
    2. **series** -- at a finite ``x0`` the expression is split into
       numerator and denominator and each is expanded in its Taylor
       coefficients ``f^(k)(x0)/k!``; the leading orders ``kn`` and ``kd``
       give ``cn/cd`` when equal, 0 when ``kn > kd``, and a signed infinity
       otherwise (the sign from the coefficients and, for odd order
       differences, from ``side``);
    3. **lhopital** -- for the indeterminate forms ``0/0`` and ``inf/inf``
       numerator and denominator are differentiated together, at most 12
       times, re-testing after each step;
    4. **order** -- at ``x0 = +-inf`` (and after the substitution ``x ->
       1/t`` for the series and L'Hopital rules) the expression is written,
       when it belongs to that class, as a sum of monomials ``c exp(P(x))
       x^a (log x)^b`` and the limit is read off the dominant one. This is
       Gruntz's most-rapidly-varying comparison restricted to exp-log
       monomials: the exp part decides first (by degree and leading
       coefficient inside ``P``), then the power of ``x``, then the power of
       ``log x``.

    Anything else raises ``ValueError`` instead of returning a guess.

    Parameters
    ----------
    expr : str
        Expression.
    x : str
        Variable.
    x0 : float or str
        Limit point; ``"inf"``, ``"+inf"``, ``"-inf"`` and the floats
        ``inf``/``-inf`` are accepted.
    side : str
        ``"both"``, ``"+"`` (from above) or ``"-"`` (from below); ignored at
        an infinite limit point.

    Returns
    -------
    RichResult
        ``limit`` (float, possibly infinite), ``method``, ``lhopital_steps``
        and ``expression``.

    References
    ----------
    Gruntz, D. (1996). *On Computing Limits in a Symbolic Manipulation
    System*. PhD thesis, ETH Zurich (the mrv comparison of chapter 3).
    Bernoulli, J. and de l'Hopital, G. F. A. (1696). *Analyse des
    infiniment petits*, section 9.
    Hardy, G. H. (1910). *Orders of Infinity*. Cambridge University Press.

    Examples
    --------
    >>> r = symbolic_limit("sin(x)/x", "x", 0)
    >>> r.limit, r.method
    (1.0, 'series')
    >>> symbolic_limit("(1 - cos(x))/x^2", "x", 0).limit
    0.5
    >>> symbolic_limit("x*exp(-x)", "x", "inf").limit
    0.0
    >>> r = symbolic_limit("(x^2 + 3*x)/(2*x^2 - 1)", "x", "inf")
    >>> r.limit, r.method
    (0.5, 'degrees')
    >>> symbolic_limit("1/x", "x", 0, side="-").limit
    -inf
    """
    if side not in ("both", "+", "-"):
        raise ValueError("side must be 'both', '+' or '-'")
    e = _parse(expr)
    if isinstance(x0, str):
        s = x0.strip().lower()
        x0 = math.inf if s in ("inf", "+inf", "infinity") else (-math.inf if s in ("-inf", "-infinity") else float(s))
    x0 = float(x0)
    if math.isinf(x0):
        sub = _simp(_subst(e, ("sym", x), _mul(_num(math.copysign(1.0, x0)), _pow(("sym", x), -1))))  # x -> +-1/t
        pos = e if x0 > 0 else _simp(_subst(e, ("sym", x), _mul(_num(-1.0), ("sym", x))))
        v = _rat_limit_inf(pos, x)
        if v is not None:
            return RichResult(payload={"limit": v, "method": "degrees", "lhopital_steps": 0, "expression": _str(e)})
        v = _asym_limit(pos, x)
        if v is not None:
            return RichResult(payload={"limit": v, "method": "order", "lhopital_steps": 0, "expression": _str(e)})
        nu0, de0 = _split_ratio(pos, x)
        if _is_bounded(nu0, x) and not _is_bounded(de0, x):
            g = _asym_limit(de0, x)
            if g is None:
                g = _rat_limit_inf(de0, x)
            if g is not None and math.isinf(g):
                return RichResult(
                    payload={"limit": 0.0, "method": "squeeze", "lhopital_steps": 0, "expression": _str(e)}
                )
        inner = symbolic_limit(_str(sub), x, 0.0, side="+")
        return RichResult(
            payload={
                "limit": inner.limit,
                "method": inner.method + "+reciprocal",
                "lhopital_steps": inner.lhopital_steps,
                "expression": _str(e),
            }
        )
    v = _num_at(e, x, x0)
    if math.isfinite(v) and _continuous_at(e, x, x0, v, side):
        return RichResult(payload={"limit": v, "method": "substitution", "lhopital_steps": 0, "expression": _str(e)})
    nu, de = _split_ratio(e, x)
    cn = _series_coefs(nu, x, x0, 8)
    cd = _series_coefs(de, x, x0, 8)
    if cn is not None and cd is not None:
        ln, ld = _lead(cn), _lead(cd)
        if ln is not None and ld is not None:
            kn, c1 = ln
            kd, c2 = ld
            if kn >= kd:
                out = c1 / c2 if kn == kd else 0.0
                return RichResult(
                    payload={"limit": out, "method": "series", "lhopital_steps": 0, "expression": _str(e)}
                )
            sgn = math.copysign(1.0, c1 / c2)
            if (kd - kn) % 2 == 1:
                if side == "both":
                    raise ValueError("the two one-sided limits differ; pass side='+' or side='-'")
                if side == "-":
                    sgn = -sgn
            return RichResult(
                payload={"limit": math.inf * sgn, "method": "series", "lhopital_steps": 0, "expression": _str(e)}
            )
        if ln is None and ld is None:
            raise ValueError("numerator and denominator vanish identically to the expanded order")
    eps = 1e-6 if side != "-" else -1e-6
    steps = 0
    while steps < _LIM_MAX:
        a = _num_at(nu, x, x0 + eps)
        b = _num_at(de, x, x0 + eps)
        if not (math.isnan(a) or math.isnan(b)):
            zero = abs(a) < 1e-4 and abs(b) < 1e-4
            big = abs(a) > 1e4 and abs(b) > 1e4
            if not (zero or big):
                break
        nu, de = _d(nu, x), _d(de, x)
        steps += 1
        w = _num_at(_simp(_mul(nu, _pow(de, -1))), x, x0)
        if math.isfinite(w):
            return RichResult(
                payload={"limit": w, "method": "lhopital", "lhopital_steps": steps, "expression": _str(e)}
            )
    raise ValueError("cannot determine this limit with substitution, series, L'Hopital or order comparison")


# ---------------------------------------------------------------- symbolic linear algebra
def _minor(rows, drop_r, drop_c):
    return [[rows[i][j] for j in range(len(rows)) if j != drop_c] for i in range(len(rows)) if i != drop_r]


def _det_expr(rows):
    n = len(rows)
    if n == 1:
        return rows[0][0]
    if n == 2:
        return _simp(_add(_mul(rows[0][0], rows[1][1]), _mul(_num(-1), _mul(rows[0][1], rows[1][0]))))
    total = _num(0)
    for j in range(n):
        if rows[0][j] == _num(0):
            continue
        sub = _det_expr(_minor(rows, 0, j))
        total = _add(total, _mul(_num(-1.0 if j % 2 else 1.0), _mul(rows[0][j], sub)))
    return _simp(total)


def matrix_symbolic(M, eigen_var: str = "t") -> RichResult:
    r"""Determinant, inverse, characteristic polynomial and eigenvalues of a matrix with symbolic entries.

    Entries are numbers or expression strings (parsed and kept in the
    canonical form of :mod:`morie.fn.symint`, so results are exact rational
    functions of the symbols, not floating-point matrices). The determinant
    is the Laplace cofactor expansion, the inverse the adjugate over the
    determinant, and the characteristic polynomial ``det(A - t I)`` is
    expanded in ``eigen_var`` by the same expansion. Eigenvalues are given
    in closed form when the characteristic polynomial is linear or
    quadratic in ``t`` (the quadratic formula on its coefficients), and
    numerically from the Durand-Kerner roots when every coefficient is a
    number; for a higher degree with symbolic coefficients only the
    polynomial is returned, since a general closed form does not exist
    (Abel-Ruffini).

    Parameters
    ----------
    M : list of lists
        Square matrix of numbers or expression strings.
    eigen_var : str
        Name of the characteristic-polynomial variable.

    Returns
    -------
    RichResult
        ``determinant``, ``inverse`` (strings, ``None`` when the determinant
        is identically zero), ``char_poly``, ``char_coeffs`` (ascending in
        ``eigen_var``, strings), ``eigenvalues`` (strings or ``None``),
        ``eigenvalues_numeric`` (floats when the entries are numeric),
        ``trace`` and ``n``.

    References
    ----------
    Horn, R. A. and Johnson, C. R. (2013). *Matrix Analysis*, 2nd edn,
    sections 0.3 (cofactors and the adjugate) and 1.2 (the characteristic
    polynomial). Cambridge University Press.
    Abel, N. H. (1824). *Memoire sur les equations algebriques*.

    Examples
    --------
    >>> r = matrix_symbolic([["a", "b"], ["c", "d"]])
    >>> r.determinant
    'a*d - b*c'
    >>> r.char_poly
    '(-a - d)*t + a*d - b*c + t^2'
    >>> matrix_symbolic([[2, 1], [1, 2]]).eigenvalues_numeric
    [1.0, 3.0]
    >>> matrix_symbolic([["x", 1], [0, "x"]]).inverse
    ['x^(-1)', '-x^(-2)', '0', 'x^(-1)']
    """
    rows = [[_parse(v) if isinstance(v, str) else _num(float(v)) for v in row] for row in M]
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError("the matrix must be square")
    if n > 6:
        raise ValueError("symbolic cofactor expansion is limited to 6 by 6")
    det = _det_expr(rows)
    T = ("sym", eigen_var)
    shifted = [[_add(rows[i][j], _mul(_num(-1), T)) if i == j else rows[i][j] for j in range(n)] for i in range(n)]
    cp = _det_expr(shifted)
    # det(A - t I) has leading coefficient (-1)^n; report the monic characteristic polynomial
    if n % 2:
        cp = _simp(_mul(_num(-1), cp))
    cc = _coefs_expr(cp, eigen_var)
    cp = _simp(("add", [_mul(c, _pow(T, k)) if k else c for k, c in enumerate(cc)]))
    coefs = _poly(cp, eigen_var)
    inv = None
    if not (det[0] == "num" and det[1] == 0.0):
        inv = []
        for i in range(n):
            for j in range(n):
                c = _det_expr(_minor(rows, j, i)) if n > 1 else _num(1)
                sign = -1.0 if (i + j) % 2 else 1.0
                inv.append(_str(_simp(_mul(_num(sign), _mul(c, _pow(det, -1))))))
    eig = None
    eig_num = None
    if coefs is not None and all(math.isfinite(c) for c in coefs):
        p = _ptrim([_snap(c) for c in coefs])
        if _pdeg(p) >= 1:
            eig_num = sorted(_snap(zr) for (zr, zi), mu in _cluster(_roots(p), p) for _ in range(mu) if abs(zi) < 1e-9)
            if len(eig_num) != _pdeg(p):
                eig_num = None
    if len(cc) == 2:
        c = cc
        eig = [_str(_simp(_mul(_num(-1), _mul(c[0], _pow(c[1], -1)))))]
    elif len(cc) == 3:
        c = cc
        disc = _simp(_add(_pow(c[1], 2), _mul(_num(-4), _mul(c[2], c[0]))))
        root = _num(math.sqrt(disc[1])) if disc[0] == "num" and disc[1] >= 0 else _pow(disc, 0.5)
        eig = [
            _str(_simp(_mul(_add(_mul(_num(-1), c[1]), _mul(_num(s), root)), _pow(_mul(_num(2), c[2]), -1))))
            for s in (1.0, -1.0)
        ]
    return RichResult(
        payload={
            "determinant": _str(det),
            "inverse": inv,
            "char_poly": _str(cp),
            "char_coeffs": [_str(c) for c in cc],
            "eigenvalues": eig,
            "eigenvalues_numeric": eig_num,
            "trace": _str(_simp(("add", [rows[i][i] for i in range(n)]))),
            "n": n,
        }
    )


def _coefs_expr(e, x):
    """Coefficients of e as a polynomial in x, as expressions (ascending), by repeated differentiation."""
    out = []
    d = e
    fact = 1.0
    for k in range(_pdeg_expr(e, x) + 1):
        if k:
            d = _d(d, x)
            fact *= k
        out.append(_simp(_mul(_num(1.0 / fact), _subst(d, ("sym", x), _num(0)))))
    return out


def _pdeg_expr(e, x, cap=8):
    """Degree of e in x, found by differentiating until the result is free of x."""
    d = e
    for k in range(cap + 1):
        if _free(d, x):
            return k
        d = _simp(_d(d, x))
    raise ValueError("the expression is not polynomial in " + x)


# ---------------------------------------------------------------- first-order ODEs
def _zero_at(e, pts):
    for env in pts:
        v = _num_at2(e, env)
        if math.isnan(v) or abs(v) > 1e-9:
            return False
    return True


def _num_at2(e, env):
    try:
        return _ev(e, env)
    except (KeyError, ZeroDivisionError, ValueError, OverflowError):
        return math.nan


_ODE_PTS = ((0.3, 0.7), (1.1, 0.4), (-0.6, 1.3), (2.2, -0.9), (0.9, 2.5))


def _expify(P):
    """exp(P) with logarithmic terms turned into powers: exp(c log u + R) = u^c exp(R)."""
    terms = P[1] if P[0] == "add" else [P]
    facs = []
    rest = []
    for t in terms:
        if t[0] == "fn" and t[1] == "log":
            facs.append(t[2])
        elif t[0] == "mul" and len(t[1]) == 2 and t[1][0][0] == "num" and t[1][1][0] == "fn" and t[1][1][1] == "log":
            facs.append(_pow(t[1][1][2], t[1][0][1]))
        else:
            rest.append(t)
    out = _simp(("mul", facs)) if facs else _num(1)
    if rest:
        out = _simp(_mul(out, _fn("exp", _simp(("add", rest)))))
    return _simp(out)


def ode_symbolic(ode=None, x: str = "x", y: str = "y", M=None, N=None) -> RichResult:
    r"""Classify a first-order ODE and solve it in closed form.

    Either ``ode``, the right-hand side of ``y' = f(x, y)`` (an equation
    whose left-hand side is ``y'`` or ``dy/dx`` is also accepted), or the
    pair ``M``, ``N`` of a differential form ``M dx + N dy = 0``. Each class
    is tested by its standard criterion, checked at five sample points:

    - **linear**: ``d2f/dy2 = 0`` with ``df/dy`` free of ``y``, so ``f =
      p(x) y + q(x)``; the integrating factor ``mu = exp(-int p dx)`` gives
      ``y = (int mu q dx + C)/mu``;
    - **bernoulli**: ``f = p(x) y + q(x) y^n`` with ``n`` neither 0 nor 1;
      ``v = y^(1-n)`` satisfies the linear equation ``v' = (1-n) p v + (1-n)
      q``, solved the same way, and ``y = v^(1/(1-n))``;
    - **separable**: ``d/dy(d/dx(log f)) = 0``, so ``f = g(x) h(y)`` with
      ``g(x) = f(x, y1)`` and ``h(y) = f(x1, y)/f(x1, y1)`` at sample values
      where ``f`` does not vanish; the solution is the implicit ``int dy/h =
      int g dx + C``;
    - **exact** (from ``M``, ``N``): ``dM/dy = dN/dx``, and the potential
      ``F = int M dx + int (N - d/dy int M dx) dy = C``;
    - **homogeneous**: ``f(t x, t y) = f(x, y)``, reported with the
      substitution ``v = y/x`` that separates it.

    The integrals come from the rules of :mod:`morie.fn.symint`; when one of
    them has no closed form the class is still reported and ``solution`` is
    ``None``.

    Parameters
    ----------
    ode : str, optional
        Right-hand side ``f(x, y)``, or ``"y' = f(x, y)"``.
    x, y : str
        Names of the independent and dependent variables.
    M, N : str, optional
        Coefficients of a differential form ``M dx + N dy = 0``.

    Returns
    -------
    RichResult
        ``classes`` (every class that applies), ``solution`` (string or
        ``None``), ``form`` (``"explicit"`` or ``"implicit"``),
        ``coefficients`` (``p`` and ``q`` for the linear and Bernoulli
        cases, ``g`` and ``h`` for the separable one), ``exponent`` (the
        Bernoulli ``n``) and ``rhs``.

    References
    ----------
    Boyce, W. E. and DiPrima, R. C. (2012). *Elementary Differential
    Equations and Boundary Value Problems*, 10th edn, sections 2.1 (linear
    equations and the integrating factor), 2.2 (separable equations), 2.4
    (the Bernoulli equation) and 2.6 (exact equations). Wiley.
    Bernoulli, J. (1695). Explicationes, annotationes et additiones.
    *Acta Eruditorum*, December 1695.

    Examples
    --------
    >>> r = ode_symbolic("y' = 2*x*y")
    >>> r.classes, r.solution
    (['separable', 'linear'], 'C*exp(x^2)')
    >>> ode_symbolic("y + x").solution
    '(C - exp(-x) - exp(-x)*x)*exp(x)'
    >>> ode_symbolic("y^2*x").form
    'implicit'
    >>> ode_symbolic(M="2*x*y", N="x^2 + 1").solution
    'x^2*y + y = C'
    """
    pts = [{x: a, y: b} for a, b in _ODE_PTS]
    if M is not None or N is not None:
        if M is None or N is None:
            raise ValueError("a differential form needs both M and N")
        Me = _parse(M)
        Ne = _parse(N)
        if not _zero_at(_simp(_add(_d(Me, y), _mul(_num(-1), _d(Ne, x)))), pts):
            return RichResult(
                payload={
                    "classes": [],
                    "solution": None,
                    "form": None,
                    "coefficients": {"M": _str(Me), "N": _str(Ne)},
                    "exponent": None,
                    "rhs": _str(_simp(_mul(_num(-1), _mul(Me, _pow(Ne, -1))))),
                }
            )
        F = _integrate(Me, x, 0)
        sol = None
        if F is not None:
            rest = _integrate(_simp(_add(Ne, _mul(_num(-1), _d(F, y)))), y, 0)
            if rest is not None:
                sol = _str(_simp(_add(F, rest))) + " = C"
        return RichResult(
            payload={
                "classes": ["exact"],
                "solution": sol,
                "form": "implicit",
                "coefficients": {"M": _str(Me), "N": _str(Ne)},
                "exponent": None,
                "rhs": _str(_simp(_mul(_num(-1), _mul(Me, _pow(Ne, -1))))),
            }
        )
    s = str(ode)
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        if lhs.strip().replace(" ", "") not in (y + "'", "d" + y + "/d" + x):
            raise ValueError("the left-hand side must be " + y + "' or d" + y + "/d" + x)
        s = rhs
    f = _parse(s)
    classes = []
    coefs = {}
    solution = None
    form = None
    n_exp = None
    fy = _simp(_d(f, y))
    fyy = _simp(_d(fy, y))
    if _zero_at(fyy, pts) and _free(fy, y):
        classes.append("linear")
        p = fy
        q = _simp(_subst(f, ("sym", y), _num(0)))
        coefs["p"] = _str(p)
        coefs["q"] = _str(q)
        P = _integrate(_simp(_mul(_num(-1), p)), x, 0)
        if P is not None:
            mu = _expify(P)
            inv = _expify(_simp(_mul(_num(-1), P)))
            inner = _integrate(_simp(_mul(mu, q)), x, 0)
            if inner is not None:
                solution = _str(_simp(_mul(_add(inner, ("sym", "C")), inv)))
                form = "explicit"
    if not classes:
        terms = f[1] if f[0] == "add" else [f]
        pw = []
        ok = True
        for term in terms:
            k = None
            rest = []
            for fac in _factors(term):
                if fac == ("sym", y):
                    k = 1.0
                elif fac[0] == "pow" and fac[1] == ("sym", y) and fac[2][0] == "num":
                    k = fac[2][1]
                elif not _free(fac, y):
                    ok = False
                else:
                    rest.append(fac)
            if not ok:
                break
            pw.append((0.0 if k is None else k, _simp(("mul", rest)) if rest else _num(1)))
        if ok and len(pw) == 2 and sorted(k for k, _ in pw) != [0.0, 1.0]:
            ks = sorted(k for k, _ in pw)
            if 1.0 in ks:
                n_exp = [k for k in ks if k != 1.0][0]
                p = next(c for k, c in pw if k == 1.0)
                q = next(c for k, c in pw if k != 1.0)
            else:
                n_exp = ks[1]
                p = _num(0)
                q = next(c for k, c in pw if k == ks[1])
            if n_exp not in (0.0, 1.0):
                classes.append("bernoulli")
                coefs["p"] = _str(p)
                coefs["q"] = _str(q)
                m = 1.0 - n_exp
                P = _integrate(_simp(_mul(_num(-m), p)), x, 0)
                if P is not None:
                    mu = _expify(P)
                    inv = _expify(_simp(_mul(_num(-1), P)))
                    inner = _integrate(_simp(_mul(mu, _mul(_num(m), q))), x, 0)
                    if inner is not None:
                        v = _simp(_mul(_add(inner, ("sym", "C")), inv))
                        solution = _str(_simp(_pow(v, 1.0 / m)))
                        form = "explicit"
    logdx = _simp(_mul(_d(f, x), _pow(f, -1)))
    if _zero_at(_simp(_d(logdx, y)), pts):
        classes.insert(0, "separable")
        g = h = None
        for y1 in (1.0, 2.0, 0.5, -1.0, 3.0):
            cand = _simp(_subst(f, ("sym", y), _num(y1)))
            if _zero_at(cand, pts):
                continue
            for x1 in (1.0, 2.0, 0.5, -1.0, 3.0):
                den = _simp(_subst(cand, ("sym", x), _num(x1)))
                scale = _num_at2(den, {})
                if math.isfinite(scale) and abs(scale) > 1e-8:
                    g = cand
                    h = _simp(_mul(_subst(f, ("sym", x), _num(x1)), _pow(den, -1)))
                    break
            if g is not None:
                break
        if g is not None:
            coefs["g"] = _str(g)
            coefs["h"] = _str(h)
            if solution is None:
                L = _integrate(_simp(_pow(h, -1)), y, 0)
                R = _integrate(g, x, 0)
                if L is not None and R is not None:
                    solution = _str(_simp(L)) + " = " + _str(_simp(_add(R, ("sym", "C"))))
                    form = "implicit"
    scaled = _simp(_subst(_subst(f, ("sym", x), _mul(_num(1.7), ("sym", x))), ("sym", y), _mul(_num(1.7), ("sym", y))))
    if _zero_at(_simp(_add(scaled, _mul(_num(-1), f))), pts):
        classes.append("homogeneous")
    if _zero_at(_simp(_d(f, y)), pts):
        classes.append("exact")
    return RichResult(
        payload={
            "classes": classes,
            "solution": solution,
            "form": form,
            "coefficients": coefs,
            "exponent": n_exp,
            "rhs": _str(f),
        }
    )


def cheatsheet() -> str:
    return (
        "symbolic_limit(expr, x, x0) -> limits by series, L'Hopital and exp-log order; "
        "matrix_symbolic(M) -> det, inverse, characteristic polynomial, eigenvalues; "
        "ode_symbolic(rhs) -> first-order ODE class and closed form; "
        "risch_integration(expr, x) -> Hermite plus Rothstein-Trager on rational integrands."
    )
