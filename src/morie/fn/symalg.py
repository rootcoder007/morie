# morie.fn -- function file (rootcoder007/morie)
"""Exact algebra: the Jordan canonical form of an integer matrix with integer spectrum, the Galois group of
an integer polynomial of degree at most four, and Dijkstra's shunting-yard parser."""

from __future__ import annotations

import math
from fractions import Fraction

from ._richresult import RichResult

__all__ = ["jordan_canonical", "galois_group", "shunting_yard"]


# ---------------------------------------------------------------- exact linear algebra
def _rref(M):
    A = [[Fraction(v) for v in r] for r in M]
    m = len(A)
    n = len(A[0]) if m else 0
    piv = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(n)]
        piv.append(c)
        r += 1
        if r == m:
            break
    return A, piv


def _rank(M):
    return len(_rref(M)[1]) if M else 0


def _primitive(v):
    den = 1
    for x in v:
        den = den * x.denominator // math.gcd(den, x.denominator)
    w = [int(x * den) for x in v]
    g = 0
    for x in w:
        g = math.gcd(g, x)
    return [x // g for x in w] if g else w


def _nullspace(M, n):
    R, piv = _rref(M)
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for i, c in enumerate(piv):
            v[c] = -R[i][f]
        basis.append(_primitive(v))
    return basis


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _charpoly(A):
    """Faddeev-LeVerrier: coefficients c_n..c_0 of det(xI - A), integers for integer A."""
    n = len(A)
    M = [[0] * n for _ in range(n)]
    c = [1]
    for k in range(1, n + 1):
        M = [[sum(A[i][t] * M[t][j] for t in range(n)) + (c[-1] if i == j else 0) for j in range(n)] for i in range(n)]
        AM = _matmul(A, M)
        c.append(-sum(AM[i][i] for i in range(n)) // k)
    return c


def _peval(c, x):
    v = 0
    for a in c:
        v = v * x + a
    return v


def _divisors(n):
    n = abs(n)
    return [d for d in range(1, n + 1) if n % d == 0]


def _pdiv_root(c, r):
    out = [c[0]]
    for a in c[1:-1]:
        out.append(a + out[-1] * r)
    return out


def _integer_roots(c):
    """Integer roots of a monic integer polynomial with multiplicity, ascending."""
    roots = []
    c = list(c)
    while len(c) > 1 and c[-1] == 0:
        roots.append(0)
        c = c[:-1]
    cands = sorted({s * d for d in _divisors(c[-1]) for s in (1, -1)}) if len(c) > 1 else []
    for r in cands:
        while len(c) > 1 and _peval(c, r) == 0:
            roots.append(r)
            c = _pdiv_root(c, r)
    return sorted(roots), c


def jordan_canonical(A) -> RichResult:
    r"""Jordan canonical form ``A = P J P^-1`` of an integer matrix whose eigenvalues are integers, in exact arithmetic.

    The characteristic polynomial comes from Faddeev-LeVerrier, its integer
    roots from the rational-root theorem. For each eigenvalue ``lambda``
    with ``N = A - lambda I``, the number of blocks of size at least ``k``
    is ``rank N^(k-1) - rank N^k``; Jordan chains ``N^(k-1) v, ..., N v, v``
    are grown from the longest blocks down, taking basis vectors of
    ``ker N^k`` independent of ``ker N^(k-1)`` and the chains already
    found. ``P`` is an integer matrix of chain vectors (primitive
    nullspace vectors), ``J`` upper bidiagonal with blocks ordered by
    eigenvalue then decreasing size. Raises ``ValueError`` when some
    eigenvalue is not an integer (i.e. not rational: a monic integer
    polynomial has only integer rational roots).

    References
    ----------
    Horn, R. A. and Johnson, C. R. (2013). *Matrix Analysis*, 2nd edn,
    section 3.1-3.2. Cambridge University Press.
    Faddeev, D. K. and Faddeeva, V. N. (1963). *Computational Methods of
    Linear Algebra*. Freeman.

    Examples
    --------
    >>> r = jordan_canonical([[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]])
    >>> r.J
    [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 4, 1], [0, 0, 0, 4]]
    >>> r.blocks
    [(1, 1), (2, 1), (4, 2)]
    """
    Am = [[int(v) for v in r] for r in A]
    n = len(Am)
    if any(len(r) != n for r in Am):
        raise ValueError("A must be square")
    if any(float(A[i][j]) != Am[i][j] for i in range(n) for j in range(n)):
        raise ValueError("A must have integer entries")
    cp = _charpoly(Am)
    roots, rest = _integer_roots(cp)
    if len(rest) > 1:
        raise ValueError("eigenvalues are not all integers; the exact Jordan form needs an algebraic extension")
    eig = sorted(set(roots))
    cols, blocks = [], []
    for lam in eig:
        mult = roots.count(lam)
        N = [[Am[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)]
        powers = [[[int(i == j) for j in range(n)] for i in range(n)]]
        ranks = [n]
        while n - ranks[-1] < mult:
            powers.append(_matmul(powers[-1], N))
            ranks.append(_rank(powers[-1]))
        m = len(ranks) - 1
        chains = []
        for k in range(m, 0, -1):
            n_ge_k = ranks[k - 1] - ranks[k]
            n_gt_k = (ranks[k] - ranks[k + 1]) if k < m else 0
            need = n_ge_k - n_gt_k
            if need == 0:
                continue
            base = _nullspace(powers[k - 1], n) if k > 1 else []
            have = []
            for ch in chains:
                have.extend(ch[: min(k, len(ch))])
            found = 0
            for v in _nullspace(powers[k], n):
                if _rank(base + have + [v]) > _rank(base + have):
                    chain = [v]
                    for _ in range(k - 1):
                        chain.insert(0, [sum(N[i][t] * chain[0][t] for t in range(n)) for i in range(n)])
                    chains.append(chain)
                    have.extend(chain)
                    found += 1
                    if found == need:
                        break
        for ch in chains:
            cols.extend(ch)
            blocks.append((lam, len(ch)))
    P = [[cols[j][i] for j in range(n)] for i in range(n)]
    J = [[0] * n for _ in range(n)]
    pos = 0
    for lam, s in blocks:
        for t in range(s):
            J[pos + t][pos + t] = lam
            if t:
                J[pos + t - 1][pos + t] = 1
        pos += s
    return RichResult(payload={"J": J, "P": P, "blocks": blocks, "charpoly": cp, "eigenvalues": eig})


# ---------------------------------------------------------------- Galois groups
def _is_square(d):
    return d >= 0 and math.isqrt(d) ** 2 == d


def _splits(disc, D):
    return _is_square(disc) or _is_square(disc * D)


def _monic(coef):
    c = [int(v) for v in coef]
    while c and c[0] == 0:
        c = c[1:]
    n = len(c) - 1
    a = c[0]
    return [1] + [c[k] * a ** (k - 1) for k in range(1, n + 1)], n


def _quad_factor(c):
    """Monic integer quartic -> (p, q, r, s) with (x^2+px+q)(x^2+rx+s), or None."""
    _, a, b, cc, d = c
    if d == 0:
        return None
    for q0 in _divisors(d):
        for q in (q0, -q0):
            s = d // q
            if s != q:
                num = cc - q * a
                if num % (s - q) == 0:
                    p = num // (s - q)
                    r = a - p
                    if q + s + p * r == b:
                        return p, q, r, s
            else:
                t = b - q - s
                disc = a * a - 4 * t
                if _is_square(disc) and (a + math.isqrt(disc)) % 2 == 0:
                    p = (a + math.isqrt(disc)) // 2
                    r = a - p
                    if p * s + q * r == cc:
                        return p, q, r, s
    return None


def _disc3(a, b, c):
    return a * a * b * b - 4 * b**3 - 4 * a**3 * c - 27 * c * c + 18 * a * b * c


def galois_group(poly) -> RichResult:
    r"""Galois group over ``Q`` of an irreducible integer polynomial of degree 1-4 (coefficients highest first).

    The polynomial is made monic by ``x -> x / a_n``. Degree 2: ``C2``.
    Degree 3: ``A3`` when the discriminant is a square, else ``S3``.
    Degree 4 (Kappe and Warren 1989): with ``f = x^4 + a x^3 + b x^2 + c x
    + d`` and resolvent cubic ``R = x^3 - b x^2 + (ac - 4d) x - (a^2 d -
    4bd + c^2)`` (same discriminant ``D`` as ``f``): ``R`` irreducible
    gives ``A4`` (``D`` square) or ``S4``; ``R`` split gives ``V4``; one
    integer root ``r`` gives ``C4`` when ``(r^2 - 4d) D`` and ``(a^2 - 4(b -
    r)) D`` are both squares, else ``D4``. Reducible input returns
    ``group = "reducible"`` with its linear roots or quadratic factors.

    References
    ----------
    Kappe, L.-C. and Warren, B. (1989). An elementary test for the Galois
    group of a quartic polynomial. *American Mathematical Monthly*, 96,
    133-137.
    Conrad, K. *Galois groups of cubics and quartics (not in characteristic
    2)*. Expository notes, University of Connecticut.

    Examples
    --------
    >>> [galois_group(p).group for p in ([1, 0, 0, 0, -2], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, -1, -1])]
    ['D4', 'V4', 'C4', 'S4']
    """
    c, n = _monic(poly)
    if n < 1 or n > 4:
        raise ValueError("degree must be between 1 and 4")
    roots, rest = _integer_roots(c)
    out = {"degree": n, "monic": c, "discriminant": None, "resolvent": None}
    if n == 1:
        out.update(group="trivial", order=1)
        return RichResult(payload=out)
    if roots:
        out.update(group="reducible", order=None, integer_roots=roots, cofactor=rest)
        return RichResult(payload=out)
    if n == 2:
        D = c[1] ** 2 - 4 * c[2]
        out.update(group="C2", order=2, discriminant=D)
        return RichResult(payload=out)
    if n == 3:
        D = _disc3(c[1], c[2], c[3])
        g = "A3" if _is_square(D) else "S3"
        out.update(group=g, order=3 if g == "A3" else 6, discriminant=D)
        return RichResult(payload=out)
    fac = _quad_factor(c)
    if fac is not None:
        out.update(group="reducible", order=None, quadratic_factors=[[1, fac[0], fac[1]], [1, fac[2], fac[3]]])
        return RichResult(payload=out)
    _, a, b, cc, d = c
    R = [1, -b, a * cc - 4 * d, -(a * a * d - 4 * b * d + cc * cc)]
    D = _disc3(R[1], R[2], R[3])
    rr, _ = _integer_roots(R)
    rr = sorted(set(rr))
    out.update(discriminant=D, resolvent=R, resolvent_roots=rr)
    if not rr:
        g = "A4" if _is_square(D) else "S4"
    elif len(rr) == 3:
        g = "V4"
    else:
        r = rr[0]
        g = "C4" if _splits(r * r - 4 * d, D) and _splits(a * a - 4 * (b - r), D) else "D4"
    out.update(group=g, order={"A4": 12, "S4": 24, "V4": 4, "C4": 4, "D4": 8}[g])
    return RichResult(payload=out)


# ---------------------------------------------------------------- shunting yard
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4, "neg": 3}
_RIGHT = {"^", "neg"}
_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}


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


def shunting_yard(tokens, variables=None) -> RichResult:
    r"""Dijkstra's shunting-yard conversion of an infix expression to reverse Polish notation, then evaluation.

    ``tokens`` is a string or a token list. Operators ``+ - * /`` (left
    associative), ``^`` (right associative, binding tighter than unary
    minus, so ``-2^2 = -4``), unary minus (``neg``), parentheses, and the
    functions ``sin cos tan exp log sqrt abs`` (one argument) plus ``min``
    and ``max`` (two, comma-separated). Operators pop from the stack while
    the top has higher precedence, or equal precedence and the incoming
    operator is left associative. The RPN is evaluated with a value stack
    when every identifier is found in ``variables``.

    References
    ----------
    Dijkstra, E. W. (1961). Algol 60 translation. *Mathematisch Centrum
    report MR 35/61*, Amsterdam.
    Aho, A. V., Lam, M. S., Sethi, R. and Ullman, J. D. (2006). *Compilers:
    Principles, Techniques, and Tools*, 2nd edn, section 2.5.

    Examples
    --------
    >>> r = shunting_yard("3 + 4 * 2 / (1 - 5) ^ 2 ^ 3")
    >>> " ".join(r.rpn), r.value
    ('3 4 2 * 1 5 - 2 3 ^ ^ / +', 3.0001220703125)
    """
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
    return RichResult(payload={"rpn": out, "value": val, "tokens": toks})


def cheatsheet() -> str:
    return (
        "jordan_canonical(A) -> exact J, P for integer spectrum; galois_group(poly) -> C2/A3/S3/V4/C4/D4/A4/S4; "
        "shunting_yard(expr) -> RPN and value."
    )
