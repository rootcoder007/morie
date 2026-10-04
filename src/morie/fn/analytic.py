# morie.fn -- function file (rootcoder007/morie)
"""Analytic and symbolic-numeric tools: numerical Laplace transform, fixed-Talbot and
Gaver-Stehfest inversion, Laurent coefficients by the trapezoidal Cauchy integral, the
orthonormal fast Walsh-Hadamard transform, polynomial expansion, Cantor-Zassenhaus
factorisation over GF(p), rational-function cancellation over the integers, the heat equation
by separation of variables, stable quadratic roots and the L2 norm of a sampled function."""

from __future__ import annotations

import cmath
import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "laplace_transform_num",
    "talbot_inverse",
    "stehfest_inverse",
    "laurent_coefficients",
    "hadamard_transform",
    "hadamard_inverse",
    "poly_expand",
    "poly_factor_mod_p",
    "rational_cancel",
    "heat_equation_series",
    "quadratic_roots",
    "functional_norm",
]


def _exp_sinh(g, h, tmax=4.0):
    n = int(round(tmax / h))
    s = 0.0
    for k in range(-n, n + 1):
        t = k * h
        u = math.exp(math.pi / 2 * math.sinh(t))
        if u == 0.0 or math.isinf(u):
            continue
        v = g(u)
        if v != 0.0:
            s += v * u * math.pi / 2 * math.cosh(t)
    return s * h


def laplace_transform_num(f, s, *, h: float = 1.0 / 128.0) -> list:
    r"""Numerical Laplace transform ``L{f}(s) = int_0^inf f(t) exp(-s t) dt`` for real ``s > 0``.

    Evaluated by the exp-sinh double-exponential rule (Takahasi and Mori
    1974), ``t = exp(pi/2 sinh(tau))``, trapezoid step ``h`` in ``tau``.

    References
    ----------
    Takahasi, H. and Mori, M. (1974). Double exponential formulas for
    numerical integration. Publ. RIMS Kyoto Univ. 9, 721-741.

    Examples
    --------
    >>> round(laplace_transform_num(lambda t: math.exp(-t), [1.0])[0], 12)
    0.5
    """
    ss = [float(v) for v in (s if isinstance(s, (list, tuple)) else [s])]
    out = []
    for sv in ss:
        out.append(_exp_sinh(lambda t, sv=sv: f(t) * math.exp(-sv * t) if sv * t < 745.0 else 0.0, h))
    return out


def talbot_inverse(F, t, *, M: int = 24) -> list:
    r"""Inverse Laplace transform by the fixed Talbot method (Abate and Valko 2004).

    With ``r = 2M / (5t)``, ``theta_k = k pi / M``,
    ``delta_0 = r``, ``delta_k = r theta_k (cot theta_k + i)``,
    ``gamma_k = 1 + i theta_k (1 + cot^2 theta_k) - i cot theta_k``:
    ``f(t) ~ (r/M) (F(r) e^(rt) / 2 + sum_{k=1}^{M-1} Re(e^(t delta_k) F(delta_k) gamma_k))``.
    ``F`` must accept complex arguments. In double precision ``M`` near 24
    is best (errors about 1e-13 for smooth ``f``); larger ``M`` loses digits.

    References
    ----------
    Abate, J. and Valko, P. P. (2004). Multi-precision Laplace transform
    inversion. Int. J. Numer. Meth. Engng 60, 979-993. Talbot, A. (1979). The
    accurate numerical inversion of Laplace transforms. IMA J. Appl. Math. 23, 97-120.

    Examples
    --------
    >>> round(talbot_inverse(lambda s: 1 / (s + 1), [1.0])[0], 12)
    0.367879441172
    """
    ts = [float(v) for v in (t if isinstance(t, (list, tuple)) else [t])]
    out = []
    for tv in ts:
        r = 2.0 * M / (5.0 * tv)
        acc = 0.5 * (F(complex(r, 0.0)) * cmath.exp(r * tv)).real
        for k in range(1, M):
            th = k * math.pi / M
            cot = math.cos(th) / math.sin(th)
            d = complex(r * th * cot, r * th)
            g = complex(1.0, th * (1.0 + cot * cot) - cot)
            acc += (cmath.exp(tv * d) * F(d) * g).real
        out.append(r / M * acc)
    return out


def _stehfest_v(N):
    h = N // 2
    V = []
    for k in range(1, N + 1):
        s = 0.0
        for j in range((k + 1) // 2, min(k, h) + 1):
            s += (
                j**h
                * math.factorial(2 * j)
                / (
                    math.factorial(h - j)
                    * math.factorial(j)
                    * math.factorial(j - 1)
                    * math.factorial(k - j)
                    * math.factorial(2 * j - k)
                )
            )
        V.append((-1) ** (k + h) * s)
    return V


def stehfest_inverse(F, t, *, N: int = 16) -> list:
    r"""Inverse Laplace transform by the Gaver-Stehfest algorithm (real ``s`` only).

    ``f(t) ~ (ln 2 / t) sum_{k=1}^{N} V_k F(k ln 2 / t)`` with
    ``V_k = (-1)^(k + N/2) sum_{j=floor((k+1)/2)}^{min(k, N/2)}
    j^(N/2) (2j)! / ((N/2 - j)! j! (j-1)! (k-j)! (2j-k)!)``, ``N`` even.
    Suits smooth non-oscillating ``f``; accuracy is limited by cancellation
    (about ``N/2`` digits lost).

    References
    ----------
    Stehfest, H. (1970). Algorithm 368: numerical inversion of Laplace
    transforms. Comm. ACM 13, 47-49.

    Examples
    --------
    >>> round(stehfest_inverse(lambda s: 1 / (s + 1), [1.0])[0], 6)
    0.367879
    """
    V = _stehfest_v(int(N))
    ts = [float(v) for v in (t if isinstance(t, (list, tuple)) else [t])]
    ln2 = math.log(2.0)
    return [ln2 / tv * ssum(V[k - 1] * F(k * ln2 / tv) for k in range(1, N + 1)) for tv in ts]


def laurent_coefficients(f, c: complex, orders, *, radius: float = 1.0, n: int = 256) -> RichResult:
    r"""Laurent-series coefficients ``a_k`` of ``f`` about ``c`` on the annulus through ``|z - c| = radius``.

    ``a_k = (1 / 2 pi i) oint f(z) (z - c)^(-k-1) dz`` by the trapezoidal
    rule on ``n`` equispaced points of the circle (spectrally accurate for
    analytic ``f`` on the annulus):
    ``a_k ~ (1/n) sum_j f(c + r w_j) (r w_j)^(-k)``, ``w_j = exp(2 pi i j / n)``.

    References
    ----------
    Laurent, P. A. (1843). Extension du theoreme de M. Cauchy. C. R. Acad.
    Sci. Paris 17, 348-349. Trefethen, L. N. and Weideman, J. A. C. (2014).
    The exponentially convergent trapezoidal rule. SIAM Review 56, 385-458.

    Examples
    --------
    >>> r = laurent_coefficients(lambda z: 1 / z + 2 + 3 * z, 0, [-1, 0, 1, 2])
    >>> [round(v, 12) + 0.0 for v in r.real]
    [1.0, 2.0, 3.0, 0.0]
    """
    c = complex(c)
    pts = [radius * cmath.exp(2j * math.pi * j / n) for j in range(n)]
    vals = [f(c + z) for z in pts]
    re, im = [], []
    for k in orders:
        s = 0j
        for z, v in zip(pts, vals):
            s += v * z ** (-k)
        a = s / n
        re.append(a.real)
        im.append(a.imag)
    return RichResult(payload={"orders": list(orders), "real": re, "imag": im})


def hadamard_transform(x) -> list:
    r"""Orthonormal fast Walsh-Hadamard transform ``y = H_d x / sqrt(d)`` (natural/Hadamard order).

    Butterfly recursion of length ``d = 2^m``. The orthonormal transform is
    its own inverse.

    References
    ----------
    Pratt, W. K., Kane, J. and Andrews, H. C. (1969). Hadamard transform
    image coding. Proc. IEEE 57, 58-68.

    Examples
    --------
    >>> hadamard_transform([1.0, 0.0, 1.0, 0.0])
    [1.0, 1.0, 0.0, 0.0]
    """
    y = [float(v) for v in x]
    d = len(y)
    if d & (d - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < d:
        for i in range(0, d, 2 * h):
            for j in range(i, i + h):
                a, b = y[j], y[j + h]
                y[j], y[j + h] = a + b, a - b
        h *= 2
    s = math.sqrt(d)
    return [v / s for v in y]


def hadamard_inverse(y) -> list:
    r"""Inverse of :func:`hadamard_transform`: ``WHT^(-1)(y) = WHT(y)`` (orthonormal, ``1/sqrt(d)``).

    Examples
    --------
    >>> hadamard_inverse(hadamard_transform([3.0, 1.0, 4.0, 1.0]))
    [3.0, 1.0, 4.0, 1.0]
    """
    return hadamard_transform(y)


def _pmul(a, b):
    if not a or not b:
        return []
    out = [0.0] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] += u * v
    return out


def poly_expand(factors, powers=None) -> list:
    r"""Expand ``prod_i f_i(x)^(e_i)`` into ascending-power coefficients.

    Examples
    --------
    >>> poly_expand([[1, 1], [-1, 1]], [2, 1])
    [-1.0, -1.0, 1.0, 1.0]
    """
    out = [1.0]
    for i, f in enumerate(factors):
        e = 1 if powers is None else int(powers[i])
        for _ in range(e):
            out = _pmul(out, [float(v) for v in f])
    return out


# ---- polynomials over GF(p): ascending integer coefficient lists, [] is zero ----


def _trim(a):
    a = list(a)
    while a and a[-1] == 0:
        a.pop()
    return a


def _inv(a, p):
    return pow(int(a), p - 2, p)


def _gsub(a, b, p):
    n = max(len(a), len(b))
    return _trim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)])


def _gmul(a, b, p):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] = (out[i + j] + u * v) % p
    return _trim(out)


def _gdivmod(a, b, p):
    a = list(a)
    q = [0] * max(len(a) - len(b) + 1, 1)
    il = _inv(b[-1], p)
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] * il % p
        q[k] = c
        for i, v in enumerate(b):
            a[i + k] = (a[i + k] - c * v) % p
        a = _trim(a)
    return _trim(q), a


def _monic(a, p):
    il = _inv(a[-1], p)
    return [v * il % p for v in a]


def _ggcd(a, b, p):
    while b:
        a, b = b, _gdivmod(a, b, p)[1]
    return _monic(a, p) if a else a


def _gpowmod(a, e, f, p):
    out, base = [1], _gdivmod(a, f, p)[1]
    while e > 0:
        if e & 1:
            out = _gdivmod(_gmul(out, base, p), f, p)[1]
        base = _gdivmod(_gmul(base, base, p), f, p)[1]
        e >>= 1
    return out


def _sff(f, p):
    # square-free factorisation of a monic f over GF(p): list of (factor, multiplicity)
    if len(f) <= 1:
        return []
    out = []
    d = _trim([(i * f[i]) % p for i in range(1, len(f))])
    if d:
        c = _ggcd(f, d, p)
        w = _gdivmod(f, c, p)[0]
        i = 1
        while len(w) > 1:
            y = _ggcd(w, c, p)
            fac = _gdivmod(w, y, p)[0]
            if len(fac) > 1:
                out.append((fac, i))
            i += 1
            w = y
            c = _gdivmod(c, y, p)[0]
        if len(c) > 1:
            root = [c[i] for i in range(0, len(c), p)]
            out += [(g, m * p) for g, m in _sff(root, p)]
    else:
        root = [f[i] for i in range(0, len(f), p)]
        out += [(g, m * p) for g, m in _sff(root, p)]
    return out


def _ddf(f, p):
    out = []
    i = 1
    fs = list(f)
    h = [0, 1]
    while len(fs) - 1 >= 2 * i:
        h = _gpowmod(h, p, fs, p)
        g = _ggcd(fs, _gsub(h, [0, 1], p), p)
        if len(g) > 1:
            out.append((g, i))
            fs = _gdivmod(fs, g, p)[0]
            h = _gdivmod(h, fs, p)[1]
        i += 1
    if len(fs) > 1:
        out.append((fs, len(fs) - 1))
    return out


def _edf(f, d, p, seed, counter):
    n = len(f) - 1
    parts = [f]
    while len(parts) < n // d:
        u = random_uniform(n, seed=seed, stream=counter[0])
        counter[0] += 1
        h = _trim([int(math.floor(float(v) * p)) for v in u])
        if len(h) < 2:
            continue
        g = [1]
        hp = h
        for j in range(d):
            if j > 0:
                hp = _gpowmod(hp, p, f, p)
            g = _gdivmod(_gmul(g, hp, p), f, p)[1]
        g = _gsub(_gpowmod(g, (p - 1) // 2, f, p), [1], p)
        new = []
        for q in parts:
            if len(q) - 1 > d:
                r = _ggcd(q, _gdivmod(g, q, p)[1], p) if g else q
                if 1 < len(r) < len(q):
                    new += [r, _gdivmod(q, r, p)[0]]
                    continue
            new.append(q)
        parts = new
    return parts


def _key(a):
    return (len(a), list(reversed(a)))


def poly_factor_mod_p(coeffs, p: int, *, seed: int = 0) -> RichResult:
    r"""Factor a polynomial over GF(p) (odd prime ``p``) by Cantor-Zassenhaus.

    ``coeffs`` are ascending-power integer coefficients. The polynomial is
    made monic (``unit`` is its leading coefficient), split square-free
    (Yun's algorithm with ``p``-th roots), then by distinct-degree
    factorisation (``gcd(f, x^(p^i) - x)``) and Cantor-Zassenhaus
    equal-degree splitting with random polynomials from the Philox stream of
    ``seed``. Factors are monic, sorted by degree then coefficients.

    References
    ----------
    Cantor, D. G. and Zassenhaus, H. (1981). A new algorithm for factoring
    polynomials over finite fields. Math. Comp. 36, 587-592. Cohen, H.
    (1996). A Course in Computational Algebraic Number Theory, section 3.4.

    Examples
    --------
    >>> r = poly_factor_mod_p([-1, 0, 0, 1], 7)
    >>> r.factors, r.multiplicities
    ([[3, 1], [5, 1], [6, 1]], [1, 1, 1])
    """
    p = int(p)
    f = _trim([int(v) % p for v in coeffs])
    unit = f[-1]
    f = _monic(f, p)
    counter = [0]
    res = []
    for g, m in _sff(f, p):
        for q, d in _ddf(g, p):
            for r in _edf(q, d, p, seed, counter) if len(q) - 1 > d else [q]:
                res.append((r, m))
    res.sort(key=lambda fm: (_key(fm[0]), fm[1]))
    return RichResult(payload={"unit": unit, "factors": [r for r, _ in res], "multiplicities": [m for _, m in res]})


# ---- integer polynomials (floats holding integers) ----


def _igcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, math.fmod(a, b)
    return a


def _content(a):
    g = 0.0
    for v in a:
        g = _igcd(g, v)
    return g


def _ztrim(a):
    a = [float(v) for v in a]
    while a and a[-1] == 0:
        a.pop()
    return a


def _prem(a, b):
    a = list(a)
    lb = b[-1]
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        la = a[-1]
        a = [v * lb for v in a]
        for i, v in enumerate(b):
            a[i + k] -= la * v
        a = _ztrim(a)
    return a


def _zdiv(a, b):
    # exact division in Z[x]
    a = list(a)
    q = [0.0] * (len(a) - len(b) + 1)
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] = c
        for i, v in enumerate(b):
            a[i + k] -= c * v
        a = _ztrim(a)
    return q


def rational_cancel(num, den) -> RichResult:
    r"""Cancel a rational function ``num(x) / den(x)`` with integer coefficients to lowest terms.

    The polynomial gcd is the primitive part of the last nonzero term of the
    primitive pseudo-remainder sequence (Collins 1967; Knuth TAOCP vol. 2,
    4.6.1); numerator and denominator are divided exactly by it, the common
    integer content is removed and the denominator's leading coefficient made
    positive. Coefficients are ascending powers and must stay below ``2^53``.

    References
    ----------
    Collins, G. E. (1967). Subresultants and reduced polynomial remainder
    sequences. J. ACM 14, 128-142. Knuth, D. E. (1998). The Art of Computer
    Programming, vol. 2, 3rd ed., section 4.6.1.

    Examples
    --------
    >>> r = rational_cancel([-1, 0, 1], [2, 2])
    >>> r.numerator, r.denominator
    ([-1.0, 1.0], [2.0])
    """
    a, b = _ztrim(num), _ztrim(den)
    for v in a + b:
        if v != math.floor(v) or abs(v) >= 2.0**53:
            raise ValueError("coefficients must be integers below 2^53")
    x, y = (a, b) if len(a) >= len(b) else (b, a)
    x = [v / _content(x) for v in x]
    y = [v / _content(y) for v in y]
    while y:
        r = _prem(x, y)
        x, y = y, ([v / _content(r) for v in r] if r else [])
    g = [v / _content(x) for v in x]
    if g[-1] < 0:
        g = [-v for v in g]
    n2, d2 = _zdiv(a, g), _zdiv(b, g)
    c = _igcd(_content(n2), _content(d2))
    n2, d2 = [v / c for v in n2], [v / c for v in d2]
    if d2[-1] < 0:
        n2, d2 = [-v for v in n2], [-v for v in d2]
    return RichResult(payload={"numerator": n2, "denominator": d2, "gcd": g})


def heat_equation_series(f, length: float, alpha: float, x, t, *, n_terms: int = 50, n_quad: int = 1000) -> RichResult:
    r"""Heat equation ``u_t = alpha u_xx`` on ``(0, L)`` with ``u(0) = u(L) = 0`` by separation of variables.

    ``u(x, t) = X(x) T(t)`` modes give
    ``u = sum_{n=1}^{N} b_n sin(n pi x / L) exp(-alpha (n pi / L)^2 t)`` with
    ``b_n = (2/L) int_0^L f(x) sin(n pi x / L) dx`` (composite Simpson,
    ``n_quad`` even panels). Returns ``u[i][j]`` at ``x_i, t_j`` and ``b``.

    References
    ----------
    Haberman, R. (2013). Applied Partial Differential Equations, 5th ed., ch. 2.

    Examples
    --------
    >>> r = heat_equation_series(lambda z: math.sin(math.pi * z), 1.0, 1.0, [0.5], [0.0, 0.1], n_terms=3)
    >>> [round(v, 12) for v in r.u[0]]
    [1.0, 0.372707838853]
    """
    L = float(length)
    m = int(n_quad) + int(n_quad) % 2
    hq = L / m
    xs = [k * hq for k in range(m + 1)]
    fv = [f(v) for v in xs]
    w = [1.0 if k in (0, m) else (4.0 if k % 2 else 2.0) for k in range(m + 1)]
    b = []
    for n in range(1, n_terms + 1):
        s = ssum(w[k] * fv[k] * math.sin(n * math.pi * xs[k] / L) for k in range(m + 1))
        b.append(2.0 / L * s * hq / 3.0)
    u = []
    for xv in x:
        row = []
        for tv in t:
            row.append(
                ssum(
                    b[n - 1] * math.sin(n * math.pi * xv / L) * math.exp(-alpha * (n * math.pi / L) ** 2 * tv)
                    for n in range(1, n_terms + 1)
                )
            )
        u.append(row)
    return RichResult(payload={"u": u, "b": b})


def quadratic_roots(a: float, b: float, c: float) -> RichResult:
    r"""Roots of ``a x^2 + b x + c = 0`` without cancellation.

    Real roots: ``q = -(b + sign(b) sqrt(b^2 - 4ac)) / 2``, ``x1 = q / a``,
    ``x2 = c / q`` (the citardauq form; Press et al. 2007, section 5.6);
    complex pair ``(-b +- i sqrt(4ac - b^2)) / (2a)`` otherwise.

    References
    ----------
    Press, W. H. et al. (2007). Numerical Recipes, 3rd ed., section 5.6.

    Examples
    --------
    >>> r = quadratic_roots(1.0, -3.0, 2.0)
    >>> r.real, r.imag
    ([2.0, 1.0], [0.0, 0.0])
    """
    disc = b * b - 4.0 * a * c
    if disc >= 0:
        sq = math.sqrt(disc)
        q = -0.5 * (b + (sq if b >= 0 else -sq))
        x1 = q / a
        x2 = c / q if q != 0 else 0.0
        return RichResult(payload={"real": [x1, x2], "imag": [0.0, 0.0], "discriminant": disc})
    im = math.sqrt(-disc) / (2.0 * a)
    return RichResult(payload={"real": [-b / (2.0 * a)] * 2, "imag": [im, -im], "discriminant": disc})


def functional_norm(t, f) -> RichResult:
    r"""L2 norm ``||f|| = sqrt(int f(t)^2 dt)`` of a sampled function and the scaled ``f / ||f||``.

    Trapezoidal rule on the (possibly uneven) grid ``t`` (Ramsay and
    Silverman 2005, the functional inner product).

    References
    ----------
    Ramsay, J. O. and Silverman, B. W. (2005). Functional Data Analysis, 2nd ed., ch. 2.

    Examples
    --------
    >>> round(functional_norm([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]).norm, 12)
    1.0
    """
    tt = [float(v) for v in t]
    ff = [float(v) for v in f]
    s = ssum((tt[i + 1] - tt[i]) * (ff[i] ** 2 + ff[i + 1] ** 2) / 2.0 for i in range(len(tt) - 1))
    nrm = math.sqrt(s)
    return RichResult(payload={"norm": nrm, "scaled": [v / nrm for v in ff]})


def cheatsheet() -> str:
    return (
        "laplace_transform_num / talbot_inverse / stehfest_inverse / laurent_coefficients / hadamard_transform / "
        "hadamard_inverse / poly_expand / poly_factor_mod_p / rational_cancel / heat_equation_series / "
        "quadratic_roots / functional_norm -> analytic and symbolic-numeric tools."
    )


# alias kept from the retired placeholder of the same name
functional_scale = functional_norm

# alias kept from the retired placeholder of the same name
inverse_laplace = talbot_inverse

# alias kept from the retired placeholder of the same name
laplace_transform = laplace_transform_num

# alias kept from the retired placeholder of the same name
laurent_series = laurent_coefficients

# alias kept from the retired placeholder of the same name
pde_separation = heat_equation_series

# alias kept from the retired placeholder of the same name
sympy_expand = poly_expand

# alias kept from the retired placeholder of the same name
sympy_factor = poly_factor_mod_p

# alias kept from the retired placeholder of the same name
sympy_simplify = rational_cancel

# alias kept from the retired placeholder of the same name
walsh_hadamard_inverse = hadamard_inverse
