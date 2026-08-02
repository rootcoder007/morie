"""morie stats core: scipy.stats subset, natively implemented.

De-numpy campaign: replaces the `from scipy import stats` surface that
morie.fn actually uses.  Distribution objects (norm, t, chi2, f, gamma,
beta, binom, poisson, uniform, expon) expose pdf/pmf, cdf, sf, ppf, isf,
plus the small set of module-level helpers in use (sem, zscore).

Numerics (classical, dependency-free):
- normal cdf via math.erf; normal ppf via the AS241-style Acklam
  rational approximation (|err| < 1.2e-9), consistent with morie's
  native RNG reference.
- regularized incomplete gamma P(a, x) by series (x < a+1) and
  continued fraction (x >= a+1)  -> gamma/chi2/poisson cdfs.
- regularized incomplete beta I_x(a, b) by Lentz continued fraction
  -> beta/t/f/binom cdfs.
- ppf for the non-normal continuous laws by bracketed Newton/bisection
  on the cdf (monotone, so globally convergent).
Every entry point is equivalence-tested against scipy in
tests/fn/test_stats_core.py.
"""

from __future__ import annotations

import math as _math


def _erf(x):
    return _math.erf(x)


def _norm_cdf(z):
    return 0.5 * (1.0 + _math.erf(z / _math.sqrt(2.0)))


def _norm_pdf(z):
    return _math.exp(-0.5 * z * z) / _math.sqrt(2.0 * _math.pi)


def _norm_ppf(p):
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -_math.inf
        if p == 1.0:
            return _math.inf
        raise ValueError("p must be in [0, 1]")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = _math.sqrt(-2 * _math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
             + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                             + b[4]) * r + 1)
    else:
        q = _math.sqrt(-2 * _math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
              + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # one Halley refinement step to push error to ~1e-15
    e = _norm_cdf(x) - p
    u = e * _math.sqrt(2 * _math.pi) * _math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _gammainc_p(a, x):
    """Regularized lower incomplete gamma P(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("invalid arguments")
    if x == 0:
        return 0.0
    ln_pre = a * _math.log(x) - x - _math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return _math.exp(ln_pre) * total
    # continued fraction for Q(a, x), Lentz
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - _math.exp(ln_pre) * h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), Lentz continued fraction."""
    if not 0.0 <= x <= 1.0 or a <= 0 or b <= 0:
        raise ValueError("invalid arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_pre = (_math.lgamma(a + b) - _math.lgamma(a) - _math.lgamma(b)
              + a * _math.log(x) + b * _math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return _math.exp(ln_pre) * _betacf(a, b, x) / a
    return 1.0 - _math.exp(ln_pre) * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _ppf_from_cdf(cdf, p, lo, hi, iters=200):
    """Monotone-cdf inversion: bisection with Newton-like midpoint bias."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return lo
        if p == 1.0:
            return hi
        raise ValueError("p must be in [0, 1]")
    # expand upper bracket if needed
    flo, fhi = cdf(lo), cdf(hi)
    grow = 0
    while fhi < p and grow < 200:
        hi *= 2.0 if hi > 0 else 0.5
        if hi == 0:
            hi = 1.0
        fhi = cdf(hi)
        grow += 1
    while flo > p and grow < 400:
        lo = lo * 2.0 if lo < 0 else lo - _bi_abs(hi)
        flo = cdf(lo)
        grow += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def _bi_abs(v):
    return v if v >= 0 else -v


def _maybe_map(fn, x):
    if isinstance(x, (list, tuple)):
        return [fn(float(v)) for v in x]
    if hasattr(x, "tolist"):
        flat = x.tolist()
        if isinstance(flat, list):
            return [fn(float(v)) for v in flat]
        return fn(float(flat))
    return fn(float(x))


class _Dist:
    """Common frozen/unfrozen scipy-like surface."""

    def __call__(self, *args, **kw):
        return self.__class__(*args, **kw)

    def sf(self, x, *args, **kw):
        c = self.cdf(x, *args, **kw)
        if isinstance(c, list):
            return [1.0 - v for v in c]
        return 1.0 - c

    def isf(self, q, *args, **kw):
        def one(v):
            return self.ppf(1.0 - v, *args, **kw)
        return _maybe_map(one, q)

    def logpdf(self, x, *args, **kw):
        return _maybe_map(lambda v: _math.log(self.pdf(v, *args, **kw)), x)


class _Norm(_Dist):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def pdf(self, x, loc=None, scale=None):
        d = self if loc is None else _Norm(loc, scale if scale is not None
                                           else 1.0)
        return _maybe_map(lambda v: _norm_pdf(d._z(v)) / d.scale, x)

    def cdf(self, x, loc=None, scale=None):
        d = self if loc is None else _Norm(loc, scale if scale is not None
                                           else 1.0)
        return _maybe_map(lambda v: _norm_cdf(d._z(v)), x)

    def ppf(self, q, loc=None, scale=None):
        d = self if loc is None else _Norm(loc, scale if scale is not None
                                           else 1.0)
        return _maybe_map(lambda v: d.loc + d.scale * _norm_ppf(v), q)

    def rvs(self, size=None, random_state=None):
        from . import _array_core as _ac
        rng = _ac.random.default_rng(random_state)
        return rng.normal(self.loc, self.scale, size)


class _Chi2(_Dist):
    def __init__(self, df=1.0):
        self.df = float(df)

    def pdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if v < 0:
                return 0.0
            if v == 0:
                return _math.inf if k < 2 else (0.5 if k == 2 else 0.0)
            ln = ((k / 2 - 1) * _math.log(v) - v / 2
                  - (k / 2) * _math.log(2) - _math.lgamma(k / 2))
            return _math.exp(ln)
        return _maybe_map(one, x)

    def cdf(self, x, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(
            lambda v: 0.0 if v <= 0 else _gammainc_p(k / 2.0, v / 2.0), x)

    def ppf(self, q, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(
            lambda v: _ppf_from_cdf(
                lambda t: 0.0 if t <= 0 else _gammainc_p(k / 2, t / 2),
                v, 0.0, k + 10.0), q)


class _T(_Dist):
    def __init__(self, df=1.0):
        self.df = float(df)

    def pdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            ln = (_math.lgamma((k + 1) / 2) - _math.lgamma(k / 2)
                  - 0.5 * _math.log(k * _math.pi)
                  - (k + 1) / 2 * _math.log1p(v * v / k))
            return _math.exp(ln)
        return _maybe_map(one, x)

    def cdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if v == 0:
                return 0.5
            ib = _betainc(k / 2.0, 0.5, k / (k + v * v))
            return 1.0 - 0.5 * ib if v > 0 else 0.5 * ib
        return _maybe_map(one, x)

    def ppf(self, q, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if v == 0.5:
                return 0.0
            if v > 0.5:                 # symmetry: invert the upper half
                return -one(1.0 - v)
            return _ppf_from_cdf(lambda u: self.cdf(u, df=k), v,
                                 -50.0 - k, 0.0)
        return _maybe_map(one, q)


class _F(_Dist):
    def __init__(self, dfn=1.0, dfd=1.0):
        self.dfn, self.dfd = float(dfn), float(dfd)

    def cdf(self, x, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)
        return _maybe_map(
            lambda v: 0.0 if v <= 0 else _betainc(
                d1 / 2.0, d2 / 2.0, d1 * v / (d1 * v + d2)), x)

    def pdf(self, x, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)

        def one(v):
            if v <= 0:
                return 0.0
            ln = (0.5 * (d1 * _math.log(d1 * v) + d2 * _math.log(d2)
                         - (d1 + d2) * _math.log(d1 * v + d2))
                  - _math.log(v)
                  - (_math.lgamma(d1 / 2) + _math.lgamma(d2 / 2)
                     - _math.lgamma((d1 + d2) / 2)))
            return _math.exp(ln)
        return _maybe_map(one, x)

    def ppf(self, q, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)
        return _maybe_map(
            lambda v: _ppf_from_cdf(lambda t: self.cdf(t, d1, d2), v,
                                    0.0, 10.0), q)


class _Gamma(_Dist):
    def __init__(self, a=1.0, loc=0.0, scale=1.0):
        self.a, self.loc, self.scale = float(a), float(loc), float(scale)

    def cdf(self, x, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)
        return _maybe_map(
            lambda v: 0.0 if v <= lo else _gammainc_p(aa, (v - lo) / sc), x)

    def pdf(self, x, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)

        def one(v):
            z = (v - lo) / sc
            if z <= 0:
                return 0.0
            ln = (aa - 1) * _math.log(z) - z - _math.lgamma(aa)
            return _math.exp(ln) / sc
        return _maybe_map(one, x)

    def ppf(self, q, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)
        return _maybe_map(
            lambda v: lo + sc * _ppf_from_cdf(
                lambda t: 0.0 if t <= 0 else _gammainc_p(aa, t),
                v, 0.0, aa + 10.0), q)


class _Beta(_Dist):
    def __init__(self, a=1.0, b=1.0):
        self.a, self.b = float(a), float(b)

    def cdf(self, x, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)
        return _maybe_map(
            lambda v: _betainc(aa, bb, min(max(v, 0.0), 1.0)), x)

    def pdf(self, x, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)

        def one(v):
            if not 0.0 < v < 1.0:
                return 0.0
            ln = (_math.lgamma(aa + bb) - _math.lgamma(aa)
                  - _math.lgamma(bb) + (aa - 1) * _math.log(v)
                  + (bb - 1) * _math.log1p(-v))
            return _math.exp(ln)
        return _maybe_map(one, x)

    def ppf(self, q, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)
        return _maybe_map(
            lambda v: _ppf_from_cdf(lambda t: _betainc(aa, bb, t), v,
                                    0.0, 1.0), q)


class _Binom(_Dist):
    def pmf(self, k, n, p):
        def one(kk):
            kk = int(round(kk))
            if kk < 0 or kk > n:
                return 0.0
            return (_math.comb(int(n), kk) * p ** kk
                    * (1.0 - p) ** (int(n) - kk))
        return _maybe_map(one, k)

    def cdf(self, k, n, p):
        def one(kk):
            kk = int(_math.floor(kk))
            if kk < 0:
                return 0.0
            if kk >= n:
                return 1.0
            # I_{1-p}(n-k, k+1)
            return _betainc(n - kk, kk + 1, 1.0 - p)
        return _maybe_map(one, k)

    def sf(self, k, n, p):
        c = self.cdf(k, n, p)
        if isinstance(c, list):
            return [1.0 - v for v in c]
        return 1.0 - c


class _Poisson(_Dist):
    def pmf(self, k, mu):
        def one(kk):
            kk = int(round(kk))
            if kk < 0:
                return 0.0
            return _math.exp(kk * _math.log(mu) - mu
                             - _math.lgamma(kk + 1)) if mu > 0 \
                else (1.0 if kk == 0 else 0.0)
        return _maybe_map(one, k)

    def cdf(self, k, mu):
        def one(kk):
            kk = int(_math.floor(kk))
            if kk < 0:
                return 0.0
            # Q(k+1, mu) regularized upper
            return 1.0 - _gammainc_p(kk + 1.0, mu)
        return _maybe_map(one, k)

    def sf(self, k, mu):
        c = self.cdf(k, mu)
        if isinstance(c, list):
            return [1.0 - v for v in c]
        return 1.0 - c


class _Uniform(_Dist):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def pdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: 1.0 / sc if lo <= v <= lo + sc else 0.0, x)

    def cdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: min(max((v - lo) / sc, 0.0), 1.0), x)

    def ppf(self, q, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: lo + sc * v, q)


class _Expon(_Dist):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def pdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: _math.exp(-(v - lo) / sc) / sc if v >= lo else 0.0, x)

    def cdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: 0.0 if v < lo else 1.0 - _math.exp(-(v - lo) / sc), x)

    def ppf(self, q, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: lo - sc * _math.log1p(-v), q)


norm = _Norm()
chi2 = _Chi2()
t = _T()
f = _F()
gamma = _Gamma()
beta = _Beta()
binom = _Binom()
poisson = _Poisson()
uniform = _Uniform()
expon = _Expon()


def sem(x, ddof=1):
    from . import _array_core as _ac
    a = _ac.asarray(x)
    return a.std(ddof=ddof) / _math.sqrt(a.size)


def zscore(x, ddof=0):
    from . import _array_core as _ac
    a = _ac.asarray(x)
    return (a - a.mean()) / a.std(ddof=ddof)
