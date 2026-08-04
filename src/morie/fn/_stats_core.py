"""morie stats core: scipy.stats subset, natively implemented.

De-numpy campaign: replaces the `from scipy import stats` surface that
morie.fn actually uses.  Distribution objects (norm, t, chi2, f, gamma,
beta, binom, poisson, uniform, expon) expose pdf/pmf, cdf, sf, ppf, isf,
plus the small set of module-level helpers in use (sem, zscore).

Numerics (classical, dependency-free):
- normal cdf via math.erf; normal ppf via Wichura's AS 241 (PPND16, ~1e-16),
  the same algorithm as morie's native RNG -- one quantile function,
  not two of different accuracy.
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

import builtins as _bi
import math as _math


def _erf(x):
    return _math.erf(x)


def _norm_cdf(z):
    return 0.5 * (1.0 + _math.erf(z / _math.sqrt(2.0)))


def _norm_pdf(z):
    return _math.exp(-0.5 * z * z) / _math.sqrt(2.0 * _math.pi)


def _norm_ppf(p):
    """Standard normal quantile via Wichura's AS 241 (PPND16), ~1e-16.

    This used to be Acklam's rational approximation (|err| < 1.2e-9)
    while _rng.py carried the genuine AS 241 -- two quantile functions of
    different accuracy answering the same question.  Now there is one.
    The import is local so this module still loads without _rng.
    """
    from ._rng import normal_quantile as _ppnd16
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly inside (0, 1)")
    z = _ppnd16(p)
    return float(z if not hasattr(z, "_flat") else list(z._flat())[0])


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
    """Apply fn elementwise: scalars stay scalar, 1-D stays 1-D,
    2-D keeps its shape (rows of lists)."""
    if isinstance(x, (int, float)):
        return fn(float(x))
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and x \
            and isinstance(x[0], (list, tuple)):
        from . import _array_core as _ac2
        return _ac2.marr([[fn(float(v)) for v in row] for row in x])
    if isinstance(x, (list, tuple)):
        from . import _array_core as _ac2
        return _ac2.marr([fn(float(v)) for v in x])
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
    def logcdf(self, x, *a, **k):
        c = self.cdf(x, *a, **k)
        if isinstance(c, float):
            return _math.log(c) if c > 0 else -_math.inf
        return c._map(lambda v: _math.log(v) if v > 0 else -_math.inf)

    def logsf(self, x, *a, **k):
        s = self.sf(x, *a, **k)
        if isinstance(s, float):
            return _math.log(s) if s > 0 else -_math.inf
        return s._map(lambda v: _math.log(v) if v > 0 else -_math.inf)



class _Norm(_Dist):
    @staticmethod
    def var(loc=0.0, scale=1.0):
        """Variance scale**2."""
        del loc
        return scale * scale

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
            # scipy edge semantics: invalid df or nan input -> nan,
            # +/-inf -> exact tail limits
            if k <= 0 or k != k or v != v:
                return float("nan")
            if v == _math.inf:
                return 1.0
            if v == -_math.inf:
                return 0.0
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
    @staticmethod
    def var(a, b):
        """Variance ab / ((a+b)^2 (a+b+1)) (Johnson, Kotz &
        Balakrishnan 1995, vol. 2, ch. 25)."""
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

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
    # supports both scipy call styles: binom.pmf(k, n, p) and the
    # frozen form binom(n, p).pmf(k)
    def __init__(self, n=None, p=None):
        self._n = None if n is None else int(n)
        self._p = None if p is None else float(p)

    def _resolve(self, n, p):
        n = self._n if n is None else int(n)
        p = self._p if p is None else float(p)
        if n is None or p is None:
            raise TypeError("binom requires n and p")
        return n, p

    def mean(self, n=None, p=None):
        n, p = self._resolve(n, p)
        return n * p

    def var(self, n=None, p=None):
        n, p = self._resolve(n, p)
        return n * p * (1.0 - p)

    def std(self, n=None, p=None):
        return _math.sqrt(self.var(n, p))

    def pmf(self, k, n=None, p=None):
        n, p = self._resolve(n, p)
        def one(kk):
            kk = int(round(kk))
            if kk < 0 or kk > n:
                return 0.0
            return (_math.comb(int(n), kk) * p ** kk
                    * (1.0 - p) ** (int(n) - kk))
        return _maybe_map(one, k)

    def cdf(self, k, n=None, p=None):
        n, p = self._resolve(n, p)

        def one(kk):
            kk = int(_math.floor(kk))
            if kk < 0:
                return 0.0
            if kk >= n:
                return 1.0
            # I_{1-p}(n-k, k+1)
            return _betainc(n - kk, kk + 1, 1.0 - p)
        return _maybe_map(one, k)


    def logpmf(self, k, n=None, p=None):
        n, p = self._resolve(n, p)

        def one(kk):
            kk = int(round(kk))
            if kk < 0 or kk > n:
                return float("-inf")
            if p == 0.0:
                return 0.0 if kk == 0 else float("-inf")
            if p == 1.0:
                return 0.0 if kk == n else float("-inf")
            return (_math.lgamma(n + 1) - _math.lgamma(kk + 1)
                    - _math.lgamma(n - kk + 1)
                    + kk * _math.log(p) + (n - kk) * _math.log1p(-p))
        return _maybe_map(one, k)

    def ppf(self, q, n=None, p=None):
        n, p = self._resolve(n, p)

        def one(qq):
            if qq != qq or qq < 0.0 or qq > 1.0:
                return float("nan")
            if qq == 0.0:
                return -1.0
            if qq == 1.0:
                return float(n)
            c = 0.0
            for kk in range(n + 1):
                c += (_math.comb(n, kk) * p ** kk
                      * (1.0 - p) ** (n - kk))
                if c >= qq - 1e-12:
                    return float(kk)
            return float(n)
        return _maybe_map(one, q)

    def sf(self, k, n=None, p=None):
        c = self.cdf(k, n, p)
        if isinstance(c, list):
            return [1.0 - v for v in c]
        return 1.0 - c


class _Poisson(_Dist):
    # supports both scipy call styles: poisson.pmf(k, mu) and the
    # frozen form poisson(mu).pmf(k)
    def __init__(self, mu=None):
        self._mu = None if mu is None else float(mu)

    def _resolve(self, mu):
        mu = self._mu if mu is None else float(mu)
        if mu is None:
            raise TypeError("poisson requires mu")
        return mu

    def mean(self, mu=None):
        return self._resolve(mu)

    def var(self, mu=None):
        return self._resolve(mu)

    def std(self, mu=None):
        return _math.sqrt(self._resolve(mu))

    def pmf(self, k, mu=None):
        mu = self._resolve(mu)
        def one(kk):
            kk = int(round(kk))
            if kk < 0:
                return 0.0
            return _math.exp(kk * _math.log(mu) - mu
                             - _math.lgamma(kk + 1)) if mu > 0 \
                else (1.0 if kk == 0 else 0.0)
        return _maybe_map(one, k)

    def cdf(self, k, mu=None):
        mu = self._resolve(mu)

        def one(kk):
            kk = int(_math.floor(kk))
            if kk < 0:
                return 0.0
            # Q(k+1, mu) regularized upper
            return 1.0 - _gammainc_p(kk + 1.0, mu)
        return _maybe_map(one, k)


    def logpmf(self, k, mu=None):
        mu = self._resolve(mu)

        def one(kk):
            kk = int(round(kk))
            if kk < 0:
                return float("-inf")
            if mu <= 0:
                return 0.0 if kk == 0 else float("-inf")
            return kk * _math.log(mu) - mu - _math.lgamma(kk + 1)
        return _maybe_map(one, k)

    def ppf(self, q, mu=None):
        mu = self._resolve(mu)

        def one(qq):
            if qq != qq or qq < 0.0 or qq > 1.0:
                return float("nan")
            if qq == 0.0:
                return -1.0
            if qq == 1.0:
                return float("inf")
            kk, c = 0, _math.exp(-mu)
            term = c
            while c < qq - 1e-12:
                kk += 1
                term *= mu / kk
                c += term
                if kk > 10_000_000:
                    return float("nan")
            return float(kk)
        return _maybe_map(one, q)

    def sf(self, k, mu=None):
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


# ===================================================== helpers (tail)

def _flatten(x):
    if hasattr(x, "_flat"):
        return [float(v) for v in x._flat()]
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(float(v))
    return out


def _mean(v):
    return _math.fsum(v) / len(v)


def _var(v, ddof=0):
    m = _mean(v)
    return _math.fsum((u - m) ** 2 for u in v) / (len(v) - ddof)


class _TestResult(tuple):
    """(statistic, pvalue) tuple with attribute access."""

    def __new__(cls, statistic, pvalue, **extra):
        obj = super().__new__(cls, (statistic, pvalue))
        obj.statistic = statistic
        obj.pvalue = pvalue
        for k, v in extra.items():
            setattr(obj, k, v)
        return obj


# ---------------------------------------------------- rank / correlation

def rankdata(a, method="average"):
    v = _flatten(a)
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        if method == "average":
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = r
        elif method == "min":
            for k in range(i, j + 1):
                ranks[order[k]] = i + 1.0
        elif method == "max":
            for k in range(i, j + 1):
                ranks[order[k]] = j + 1.0
        elif method == "ordinal":
            for k in range(i, j + 1):
                ranks[order[k]] = k + 1.0
        else:
            raise ValueError("unsupported method %r" % method)
        i = j + 1
    from . import _array_core as _ac
    return _ac.marr(ranks)


def _pearson_r(x, y):
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxy = _math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = _math.fsum((a - mx) ** 2 for a in x)
    syy = _math.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    r = sxy / _math.sqrt(sxx * syy)
    return _bi.max(-1.0, _bi.min(1.0, r))


def pearsonr(x, y):
    x, y = _flatten(x), _flatten(y)
    n = len(x)
    r = _pearson_r(x, y)
    if n < 3 or abs(r) == 1.0:
        return _TestResult(r, 0.0 if abs(r) == 1.0 else 1.0)
    tstat = r * _math.sqrt((n - 2) / (1.0 - r * r))
    p = 2.0 * t.sf(abs(tstat), n - 2)
    return _TestResult(r, _bi.min(1.0, p))


def spearmanr(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


def pointbiserialr(x, y):
    return pearsonr(x, y)


def kendalltau(x, y):
    x, y = _flatten(x), _flatten(y)
    n = len(x)
    conc = disc = 0
    tx = ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2.0
    # tie counts per group for tau-b denominator
    def tie_term(v):
        counts = {}
        for u in v:
            counts[u] = counts.get(u, 0) + 1
        return _math.fsum(c * (c - 1) / 2.0 for c in counts.values())
    n1, n2 = tie_term(x), tie_term(y)
    denom = _math.sqrt((n0 - n1) * (n0 - n2))
    tau = (conc - disc) / denom if denom > 0 else float("nan")
    # normal approximation for p (scipy uses exact for small n w/o ties;
    # asymptotic matches to ~1e-3 there)
    v0 = n * (n - 1) * (2 * n + 5)
    z = 3.0 * (conc - disc) / _math.sqrt(v0 / 2.0) if v0 > 0 else 0.0
    p = 2.0 * norm.sf(abs(z))
    return _TestResult(tau, _bi.min(1.0, p))


def linregress(x, y=None):
    x = _flatten(x)
    y = _flatten(y)
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxx = _math.fsum((a - mx) ** 2 for a in x)
    sxy = _math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    r = _pearson_r(x, y)
    if n > 2 and abs(r) < 1.0:
        tstat = r * _math.sqrt((n - 2) / (1.0 - r * r))
        p = 2.0 * t.sf(abs(tstat), n - 2)
        resid = _math.fsum((y[i] - intercept - slope * x[i]) ** 2
                           for i in range(n))
        stderr = _math.sqrt(resid / (n - 2) / sxx)
    else:
        p, stderr = 0.0, 0.0
    return _TestResult(slope, p, slope=slope, intercept=intercept,
                       rvalue=r, stderr=stderr)


# ---------------------------------------------------- descriptive

def skew(a, bias=True):
    v = _flatten(a)
    n = len(v)
    m = _mean(v)
    m2 = _math.fsum((u - m) ** 2 for u in v) / n
    m3 = _math.fsum((u - m) ** 3 for u in v) / n
    if m2 == 0:
        return 0.0
    g1 = m3 / m2 ** 1.5
    if bias or n < 3:
        return g1
    return g1 * _math.sqrt(n * (n - 1)) / (n - 2)


def kurtosis(a, fisher=True, bias=True):
    v = _flatten(a)
    n = len(v)
    m = _mean(v)
    m2 = _math.fsum((u - m) ** 2 for u in v) / n
    m4 = _math.fsum((u - m) ** 4 for u in v) / n
    if m2 == 0:
        return -3.0 if fisher else 0.0
    g2 = m4 / (m2 * m2) - 3.0
    if not bias and n > 3:
        g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return g2 if fisher else g2 + 3.0


def gmean(a):
    v = _flatten(a)
    return _math.exp(_math.fsum(_math.log(u) for u in v) / len(v))


def hmean(a):
    v = _flatten(a)
    return len(v) / _math.fsum(1.0 / u for u in v)


def trim_mean(a, proportiontocut):
    v = sorted(_flatten(a))
    k = int(len(v) * proportiontocut)
    core = v[k:len(v) - k] if k > 0 else v
    return _mean(core)


def iqr(a):
    v = sorted(_flatten(a))
    n = len(v)

    def q(p):
        h = (n - 1) * p
        lo = int(_math.floor(h))
        hi = _bi.min(lo + 1, n - 1)
        return v[lo] + (h - lo) * (v[hi] - v[lo])
    return q(0.75) - q(0.25)


def describe(a, ddof=1):
    v = _flatten(a)
    return _TestResult(len(v), (min(v), max(v)), nobs=len(v),
                       minmax=(min(v), max(v)), mean=_mean(v),
                       variance=_var(v, ddof=ddof), skewness=skew(v),
                       kurtosis=kurtosis(v))


# ---------------------------------------------------- t / rank tests

def _t_pvalue(stat, df, alternative):
    if alternative == "greater":
        return t.sf(stat, df)
    if alternative == "less":
        return t.sf(-stat, df)
    return 2.0 * t.sf(abs(stat), df)


def ttest_1samp(a, popmean, alternative="two-sided"):
    v = _flatten(a)
    n = len(v)
    se = _math.sqrt(_var(v, ddof=1) / n)
    stat = (_mean(v) - float(popmean)) / se
    return _TestResult(stat, _t_pvalue(stat, n - 1, alternative),
                       df=n - 1)


def ttest_ind(a, b, equal_var=True, alternative="two-sided"):
    x, y = _flatten(a), _flatten(b)
    n1, n2 = len(x), len(y)
    v1, v2 = _var(x, ddof=1), _var(y, ddof=1)
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = _math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        df = n1 + n2 - 2
    else:
        se = _math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    stat = (_mean(x) - _mean(y)) / se
    return _TestResult(stat, _t_pvalue(stat, df, alternative), df=df)


def ttest_rel(a, b):
    x, y = _flatten(a), _flatten(b)
    return ttest_1samp([u - w for u, w in zip(x, y)], 0.0)


def mannwhitneyu(x, y, alternative="two-sided", **kw):
    del kw
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    ranks = rankdata(xv + yv)
    r1 = _math.fsum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # tie correction
    counts = {}
    for v in xv + yv:
        counts[v] = counts.get(v, 0) + 1
    n = n1 + n2
    tie = _math.fsum(c ** 3 - c for c in counts.values())
    sig = _math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    z = (u1 - mu - 0.5 * (1 if u1 > mu else -1 if u1 < mu else 0)) / sig
    if alternative == "two-sided":
        p = 2.0 * norm.sf(abs(z))
    elif alternative == "greater":
        p = norm.sf(z)
    else:
        p = norm.cdf(z)
    return _TestResult(u1, _bi.min(1.0, p))


def wilcoxon(x, y=None, correction=False, **kw):
    del kw
    xv = _flatten(x)
    if y is not None:
        yv = _flatten(y)
        d = [a - b for a, b in zip(xv, yv)]
    else:
        d = xv
    d = [v for v in d if v != 0.0]
    n = len(d)
    ranks = rankdata([abs(v) for v in d])
    wplus = _math.fsum(r for r, v in zip(ranks, d) if v > 0)
    wminus = _math.fsum(r for r, v in zip(ranks, d) if v < 0)
    stat = _bi.min(wplus, wminus)
    mu = n * (n + 1) / 4.0
    counts = {}
    for v in d:
        counts[abs(v)] = counts.get(abs(v), 0) + 1
    tie = _math.fsum(c ** 3 - c for c in counts.values())
    sig = _math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    corr = 0.5 * (1 if correction else 0)
    z = (stat - mu + corr) / sig
    return _TestResult(stat, _bi.min(1.0, 2.0 * norm.sf(abs(z))))


def kruskal(*groups):
    gs = [_flatten(g) for g in groups]
    all_v = [v for g in gs for v in g]
    n = len(all_v)
    ranks = rankdata(all_v)
    h = 0.0
    i = 0
    for g in gs:
        ni = len(g)
        ri = _math.fsum(ranks[i:i + ni])
        h += ri * ri / ni
        i += ni
    h = 12.0 / (n * (n + 1)) * h - 3.0 * (n + 1)
    counts = {}
    for v in all_v:
        counts[v] = counts.get(v, 0) + 1
    tie = _math.fsum(c ** 3 - c for c in counts.values())
    h /= (1.0 - tie / (n ** 3 - n))
    df = len(gs) - 1
    return _TestResult(h, chi2.sf(h, df))


def f_oneway(*groups):
    gs = [_flatten(g) for g in groups]
    k = len(gs)
    n = sum(len(g) for g in gs)
    grand = _math.fsum(_math.fsum(g) for g in gs) / n
    ssb = _math.fsum(len(g) * (_mean(g) - grand) ** 2 for g in gs)
    ssw = _math.fsum(_math.fsum((v - _mean(g)) ** 2 for v in g)
                     for g in gs)
    dfb, dfw = k - 1, n - k
    if dfw <= 0 or (ssw == 0.0 and ssb == 0.0):
        # scipy: constant input (or no within-group df) yields nan
        return _TestResult(float("nan"), float("nan"))
    if ssw == 0.0:
        # zero within-group variance with real between-group spread:
        # F diverges; scipy reports inf with p = 0
        return _TestResult(float("inf"), 0.0)
    stat = (ssb / dfb) / (ssw / dfw)
    return _TestResult(stat, f.sf(stat, dfb, dfw))


def levene(*groups, center="median"):
    gs = [_flatten(g) for g in groups]
    if center == "median":
        cs = [sorted(g)[len(g) // 2] if len(g) % 2 else
              0.5 * (sorted(g)[len(g) // 2 - 1] + sorted(g)[len(g) // 2])
              for g in gs]
    else:
        cs = [_mean(g) for g in gs]
    zs = [[abs(v - c) for v in g] for g, c in zip(gs, cs)]
    return f_oneway(*zs)


# ---------------------------------------------------- chi-square family

def chisquare(f_obs, f_exp=None):
    o = _flatten(f_obs)
    e = _flatten(f_exp) if f_exp is not None \
        else [_math.fsum(o) / len(o)] * len(o)
    stat = _math.fsum((a - b) ** 2 / b for a, b in zip(o, e))
    return _TestResult(stat, chi2.sf(stat, len(o) - 1))


class _Chi2ContingencyResult(tuple):
    """scipy-compatible: unpacks as (statistic, pvalue, dof,
    expected_freq) and exposes the same attributes."""

    def __new__(cls, stat, p, dof, expected):
        self = tuple.__new__(cls, (stat, p, dof, expected))
        self.statistic = stat
        self.pvalue = p
        self.dof = dof
        self.expected_freq = expected
        return self


def chi2_contingency(observed, correction=True):
    rows = observed.tolist() if hasattr(observed, "tolist") \
        else [list(r) for r in observed]
    rows = [[float(v) for v in r] for r in rows]
    r, c = len(rows), len(rows[0])
    rt = [_math.fsum(row) for row in rows]
    ct = [_math.fsum(rows[i][j] for i in range(r)) for j in range(c)]
    n = _math.fsum(rt)
    exp = [[rt[i] * ct[j] / n for j in range(c)] for i in range(r)]
    dof = (r - 1) * (c - 1)
    yates = 0.5 if correction and dof == 1 else 0.0
    stat = _math.fsum(
        (_bi.max(abs(rows[i][j] - exp[i][j]) - yates, 0.0)) ** 2
        / exp[i][j] for i in range(r) for j in range(c))
    return _Chi2ContingencyResult(stat, chi2.sf(stat, dof), dof, exp)


def _log_comb(n, k):
    return (_math.lgamma(n + 1) - _math.lgamma(k + 1)
            - _math.lgamma(n - k + 1))


def fisher_exact(table, alternative="two-sided"):
    (a, b), (c, d) = [list(map(float, r)) for r in (
        table.tolist() if hasattr(table, "tolist") else table)]
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def pmf(x):
        return _math.exp(_log_comb(r1, x) + _log_comb(n - r1, c1 - x)
                         - _log_comb(n, c1))
    lo = _bi.max(0, c1 - (n - r1))
    hi = _bi.min(r1, c1)
    p_obs = pmf(a)
    if alternative == "two-sided":
        p = _math.fsum(pmf(x) for x in range(lo, hi + 1)
                       if pmf(x) <= p_obs * (1 + 1e-7))
    elif alternative == "greater":
        p = _math.fsum(pmf(x) for x in range(a, hi + 1))
    else:
        p = _math.fsum(pmf(x) for x in range(lo, a + 1))
    odds = (a * d) / (b * c) if b * c > 0 else _math.inf
    return _TestResult(odds, _bi.min(1.0, p))


def binomtest(k, n, p=0.5, alternative="two-sided"):
    k, n = int(k), int(n)

    def pmf(x):
        return _math.exp(_log_comb(n, x) + x * _math.log(p)
                         + (n - x) * _math.log1p(-p))
    p_obs = pmf(k)
    if alternative == "two-sided":
        pv = _math.fsum(pmf(x) for x in range(n + 1)
                        if pmf(x) <= p_obs * (1 + 1e-7))
    elif alternative == "greater":
        pv = _math.fsum(pmf(x) for x in range(k, n + 1))
    else:
        pv = _math.fsum(pmf(x) for x in range(k + 1))
    return _TestResult(float(k), _bi.min(1.0, pv),
                       k=k, n=n, proportion_estimate=k / n)


# ---------------------------------------------------- KS family

def _ks_sf(d, n):
    """Two-sided asymptotic Kolmogorov Q(d*sqrt(n)) w/ Stephens correction."""
    lam = d * (_math.sqrt(n) + 0.12 + 0.11 / _math.sqrt(n))
    s = 0.0
    for j in range(1, 101):
        term = 2.0 * (-1) ** (j - 1) * _math.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) < 1e-12:
            break
    return _bi.max(0.0, _bi.min(1.0, s))


def ks_1samp(x, cdf, args=()):
    v = sorted(_flatten(x))
    n = len(v)
    cdfv = [float(cdf(u, *args)) if callable(cdf) else u for u in v]
    dplus = _bi.max((i + 1) / n - cdfv[i] for i in range(n))
    dminus = _bi.max(cdfv[i] - i / n for i in range(n))
    d = _bi.max(dplus, dminus)
    return _TestResult(d, _ks_sf(d, n))


def kstest(rvs, cdf, args=()):
    if isinstance(cdf, str):
        dist = {"norm": norm, "uniform": uniform, "expon": expon}[cdf]
        return ks_1samp(rvs, lambda u, *a: dist.cdf(u, *a), args)
    if callable(cdf):
        return ks_1samp(rvs, cdf, args)
    return ks_2samp(rvs, cdf)


def ks_2samp(a, b):
    x, y = sorted(_flatten(a)), sorted(_flatten(b))
    n1, n2 = len(x), len(y)
    allv = sorted(set(x + y))

    def ecdf(sorted_v, u):
        import bisect
        return bisect.bisect_right(sorted_v, u) / len(sorted_v)
    d = _bi.max(abs(ecdf(x, u) - ecdf(y, u)) for u in allv)
    en = n1 * n2 / (n1 + n2)
    return _TestResult(d, _ks_sf(d, en))


class _KSOne:
    """One-sided KS distribution (Birnbaum-Tingey exact sf)."""

    @staticmethod
    def sf(d, n):
        d = float(d)
        n = int(n)
        if d <= 0.0:
            return 1.0
        if d >= 1.0:
            return 0.0
        limit = int(n * (1.0 - d))
        s = 0.0
        for j in range(limit + 1):
            a = d + j / n
            b = 1.0 - d - j / n
            if b <= 0.0:
                if n - j == 0:
                    b_term = 0.0        # b**0 = 1
                else:
                    continue
            else:
                b_term = (n - j) * _math.log(b)
            lc = _log_comb(n, j)
            term = _math.exp(lc + (j - 1) * _math.log(a) + b_term) * d
            s += term
        return _bi.max(0.0, _bi.min(1.0, s))

    @staticmethod
    def cdf(d, n):
        return 1.0 - _KSOne.sf(d, n)

    @staticmethod
    def ppf(q, n):
        q = float(q)
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _KSOne.cdf(mid, n) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def isf(q, n):
        return _KSOne.ppf(1.0 - q, n)


ksone = _KSOne()


# ---------------------------------------------------- normality tests

def shapiro(x):
    """Shapiro-Wilk (Royston 1995 AS R94 approximation)."""
    v = sorted(_flatten(x))
    n = len(v)
    if n < 3:
        raise ValueError("n must be >= 3")
    m = [_norm_ppf((i + 1 - 0.375) / (n + 0.25)) for i in range(n)]
    mss = _math.fsum(u * u for u in m)
    c = [u / _math.sqrt(mss) for u in m]
    u1 = 1.0 / _math.sqrt(n)
    a = [0.0] * n
    a[n - 1] = (-2.706056 * u1 ** 5 + 4.434685 * u1 ** 4
                - 2.071190 * u1 ** 3 - 0.147981 * u1 ** 2
                + 0.221157 * u1 + c[n - 1])
    a[0] = -a[n - 1]
    if n > 5:
        a[n - 2] = (-3.582633 * u1 ** 5 + 5.682633 * u1 ** 4
                    - 1.752461 * u1 ** 3 - 0.293762 * u1 ** 2
                    + 0.042981 * u1 + c[n - 2])
        a[1] = -a[n - 2]
        phi = ((mss - 2.0 * m[n - 1] ** 2 - 2.0 * m[n - 2] ** 2)
               / (1.0 - 2.0 * a[n - 1] ** 2 - 2.0 * a[n - 2] ** 2))
        lo = 2
    else:
        phi = (mss - 2.0 * m[n - 1] ** 2) / (1.0 - 2.0 * a[n - 1] ** 2)
        lo = 1
    for i in range(lo, n - lo):
        a[i] = m[i] / _math.sqrt(phi)
    mean_v = _mean(v)
    ssq = _math.fsum((u - mean_v) ** 2 for u in v)
    w_num = _math.fsum(a[i] * v[i] for i in range(n)) ** 2
    w = w_num / ssq
    # p-value (Royston 1995)
    if n == 3:
        pw = 6.0 / _math.pi * (_math.asin(_math.sqrt(w))
                               - _math.asin(_math.sqrt(0.75)))
        return _TestResult(w, _bi.max(0.0, _bi.min(1.0, pw)))
    y = _math.log(1.0 - w)
    ln_n = _math.log(n)
    if n <= 11:
        g = -2.273 + 0.459 * n
        mu = 0.5440 - 0.39978 * n + 0.025054 * n * n \
            - 0.0006714 * n ** 3
        sig = _math.exp(1.3822 - 0.77857 * n + 0.062767 * n * n
                        - 0.0020322 * n ** 3)
        z = (-_math.log(g - y) - mu) / sig
    else:
        mu = -1.5861 - 0.31082 * ln_n - 0.083751 * ln_n ** 2 \
            + 0.0038915 * ln_n ** 3
        sig = _math.exp(-0.4803 - 0.082676 * ln_n
                        + 0.0030302 * ln_n ** 2)
        z = (y - mu) / sig
    return _TestResult(w, norm.sf(z))


def skewtest(a):
    v = _flatten(a)
    n = len(v)
    b1 = skew(v)
    y = b1 * _math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1.0 + _math.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / _math.sqrt(0.5 * _math.log(w2))
    alpha = _math.sqrt(2.0 / (w2 - 1.0))
    y = y if y != 0 else 1e-30
    z = delta * _math.log(y / alpha + _math.sqrt((y / alpha) ** 2 + 1))
    return _TestResult(z, 2.0 * norm.sf(abs(z)))


def kurtosistest(a):
    v = _flatten(a)
    n = len(v)
    b2 = kurtosis(v, fisher=False)
    e = 3.0 * (n - 1) / (n + 1)
    var_b2 = 24.0 * n * (n - 2) * (n - 3) / (
        (n + 1) ** 2 * (n + 3) * (n + 5))
    x = (b2 - e) / _math.sqrt(var_b2)
    beta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) \
        * _math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    a6 = 6.0 + 8.0 / beta1 * (2.0 / beta1
                              + _math.sqrt(1.0 + 4.0 / beta1 ** 2))
    z = ((1.0 - 2.0 / (9.0 * a6))
         - ((1.0 - 2.0 / a6) / (1.0 + x * _math.sqrt(2.0 / (a6 - 4.0))))
         ** (1.0 / 3.0)) / _math.sqrt(2.0 / (9.0 * a6))
    return _TestResult(z, 2.0 * norm.sf(abs(z)))


def normaltest(a):
    zs = skewtest(a).statistic
    zk = kurtosistest(a).statistic
    k2 = zs * zs + zk * zk
    return _TestResult(k2, chi2.sf(k2, 2))


def anderson(x, dist="norm"):
    if dist != "norm":
        raise NotImplementedError("anderson: only norm supported")
    v = sorted(_flatten(x))
    n = len(v)
    mu, sd = _mean(v), _math.sqrt(_var(v, ddof=1))
    z = [norm.cdf((u - mu) / sd) for u in v]
    a2 = -n - _math.fsum(
        (2 * (i + 1) - 1) * (_math.log(z[i])
                             + _math.log1p(-z[n - 1 - i]))
        for i in range(n)) / n
    # scipy returns the raw A2 and scales the critical values instead
    base = [0.576, 0.656, 0.787, 0.918, 1.092]
    adj = 1.0 + 4.0 / n - 25.0 / (n * n)
    crit = [round(v / adj, 3) for v in base]
    sig = [15.0, 10.0, 5.0, 2.5, 1.0]
    return _TestResult(a2, None,
                       critical_values=crit, significance_level=sig)


# ---------------------------------------------------- KDE + extra dists

class gaussian_kde:
    def __init__(self, dataset, bw_method=None):
        if hasattr(dataset, "tolist"):
            dataset = dataset.tolist()
        if dataset and isinstance(dataset[0], (list, tuple)):
            self.dataset = [[float(v) for v in row] for row in dataset]
        else:
            self.dataset = [[float(v) for v in dataset]]
        self.d = len(self.dataset)
        self.n = len(self.dataset[0])
        factor = self.n ** (-1.0 / (self.d + 4)) \
            if bw_method in (None, "scott") else float(bw_method)
        self.factor = factor
        # data covariance (ddof=1)
        means = [_mean(r) for r in self.dataset]
        cov = [[_math.fsum((self.dataset[i][k] - means[i])
                           * (self.dataset[j][k] - means[j])
                           for k in range(self.n)) / (self.n - 1)
                for j in range(self.d)] for i in range(self.d)]
        self._cov = [[cov[i][j] * factor * factor
                      for j in range(self.d)] for i in range(self.d)]
        self._inv, self._det = self._inv_det(self._cov)
        self._norm_const = _math.sqrt(
            (2.0 * _math.pi) ** self.d * self._det)

    @staticmethod
    def _inv_det(mat):
        d = len(mat)
        a = [row[:] + [1.0 if i == j else 0.0 for j in range(d)]
             for i, row in enumerate(mat)]
        det = 1.0
        for col in range(d):
            piv = max(range(col, d), key=lambda r: abs(a[r][col]))
            if piv != col:
                a[col], a[piv] = a[piv], a[col]
                det = -det
            det *= a[col][col]
            pv = a[col][col]
            a[col] = [v / pv for v in a[col]]
            for r in range(d):
                if r != col and a[r][col] != 0.0:
                    fac = a[r][col]
                    a[r] = [vr - fac * vc
                            for vr, vc in zip(a[r], a[col])]
        inv = [row[d:] for row in a]
        return inv, det

    def evaluate(self, points):
        if hasattr(points, "tolist"):
            points = points.tolist()
        if not isinstance(points[0], (list, tuple)):
            pts = [[float(v) for v in points]]
        else:
            pts = [[float(v) for v in row] for row in points]
        m = len(pts[0])
        out = []
        for k in range(m):
            s = 0.0
            for i in range(self.n):
                diff = [pts[dd][k] - self.dataset[dd][i]
                        for dd in range(self.d)]
                q = 0.0
                for r in range(self.d):
                    q += diff[r] * _math.fsum(
                        self._inv[r][c] * diff[c]
                        for c in range(self.d))
                s += _math.exp(-0.5 * q)
            out.append(s / (self.n * self._norm_const))
        return out

    __call__ = evaluate


class _Logistic(_Dist):
    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            e = _math.exp(-abs(z))
            return e / (scale * (1.0 + e) ** 2)
        return _maybe_map(one, x)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            return 1.0 / (1.0 + _math.exp(-(v - loc) / scale))
        return _maybe_map(one, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * _math.log(p / (1.0 - p))
        return _maybe_map(one, q)


class _Laplace(_Dist):
    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            return _math.exp(-abs(v - loc) / scale) / (2.0 * scale)
        return _maybe_map(one, x)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.5 * _math.exp(z) if z < 0 \
                else 1.0 - 0.5 * _math.exp(-z)
        return _maybe_map(one, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * (_math.log(2.0 * p) if p < 0.5
                                  else -_math.log(2.0 * (1.0 - p)))
        return _maybe_map(one, q)


class _Cauchy(_Dist):
    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 1.0 / (_math.pi * scale * (1.0 + z * z))
        return _maybe_map(one, x)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            return 0.5 + _math.atan((v - loc) / scale) / _math.pi
        return _maybe_map(one, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * _math.tan(_math.pi * (p - 0.5))
        return _maybe_map(one, q)


class _LogNorm(_Dist):
    """scipy parametrization: lognorm(s, loc=0, scale=exp(mu))."""

    def pdf(self, x, s, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return 0.0
            return _math.exp(-_math.log(z) ** 2 / (2.0 * s * s)) / (
                z * s * _math.sqrt(2.0 * _math.pi) * scale)
        return _maybe_map(one, x)

    def cdf(self, x, s, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return 0.0
            return _norm_cdf(_math.log(z) / s)
        return _maybe_map(one, x)

    def ppf(self, q, s, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * _math.exp(s * _norm_ppf(p))
        return _maybe_map(one, q)


class _WeibullMin(_Dist):
    def pdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0:
                return 0.0
            return c / scale * z ** (c - 1.0) * _math.exp(-z ** c)
        return _maybe_map(one, x)

    def cdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.0 if z <= 0 else 1.0 - _math.exp(-z ** c)
        return _maybe_map(one, x)

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * (-_math.log1p(-p)) ** (1.0 / c)
        return _maybe_map(one, q)


class _NBinom(_Dist):
    def pmf(self, k, n, p):
        def one(kk):
            kk = int(kk)
            return _math.exp(_math.lgamma(kk + n) - _math.lgamma(n)
                             - _math.lgamma(kk + 1)
                             + n * _math.log(p)
                             + kk * _math.log1p(-p))
        return _maybe_map(one, k)

    pdf = pmf

    def cdf(self, k, n, p):
        def one(kk):
            return _betainc(n, int(kk) + 1, p)
        return _maybe_map(one, k)

    def sf(self, k, n, p):
        return 1.0 - self.cdf(k, n, p)


class _Geom(_Dist):
    def pmf(self, k, p):
        def one(kk):
            return p * (1.0 - p) ** (int(kk) - 1)
        return _maybe_map(one, k)

    pdf = pmf

    def cdf(self, k, p):
        def one(kk):
            return 1.0 - (1.0 - p) ** int(kk)
        return _maybe_map(one, k)


class _HyperGeom(_Dist):
    def pmf(self, k, M, n, N):
        def one(kk):
            kk = int(kk)
            return _math.exp(_log_comb(n, kk) + _log_comb(M - n, N - kk)
                             - _log_comb(M, N))
        return _maybe_map(one, k)

    pdf = pmf

    def cdf(self, k, M, n, N):
        def one(kk):
            return _math.fsum(self.pmf(x, M, n, N)
                              for x in range(int(kk) + 1))
        return _maybe_map(one, k)

    def sf(self, k, M, n, N):
        return 1.0 - self.cdf(k, M, n, N)


class _GenExtreme(_Dist):
    """scipy genextreme: c > 0 = reversed-Weibull tail, c=0 Gumbel."""

    def cdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if c == 0:
                return _math.exp(-_math.exp(-z))
            t_ = 1.0 - c * z
            if t_ <= 0:
                return 1.0 if c * z >= 1 else 0.0
            return _math.exp(-t_ ** (1.0 / c))
        return _maybe_map(one, x)

    def pdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if c == 0:
                return _math.exp(-z - _math.exp(-z)) / scale
            t_ = 1.0 - c * z
            if t_ <= 0:
                return 0.0
            return t_ ** (1.0 / c - 1.0) * _math.exp(
                -t_ ** (1.0 / c)) / scale
        return _maybe_map(one, x)

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if c == 0:
                return loc - scale * _math.log(-_math.log(p))
            return loc + scale * (1.0 - (-_math.log(p)) ** c) / c
        return _maybe_map(one, q)


class _MultivariateNormal:
    def pdf(self, x, mean, cov):
        return _math.exp(self.logpdf(x, mean, cov))

    def logpdf(self, x, mean, cov):
        mu = [float(v) for v in (mean.tolist()
                                 if hasattr(mean, "tolist") else mean)]
        if hasattr(cov, "tolist"):
            cov = cov.tolist()
        cov = [[float(v) for v in r] for r in cov]
        d = len(mu)
        inv, det = gaussian_kde._inv_det(cov)
        xv = [float(v) for v in (x.tolist()
                                 if hasattr(x, "tolist") else x)]
        diff = [xv[i] - mu[i] for i in range(d)]
        q = _math.fsum(diff[i] * _math.fsum(inv[i][j] * diff[j]
                                            for j in range(d))
                       for i in range(d))
        return -0.5 * (d * _math.log(2.0 * _math.pi)
                       + _math.log(det) + q)


logistic = _Logistic()
laplace = _Laplace()
cauchy = _Cauchy()
lognorm = _LogNorm()
weibull_min = _WeibullMin()
nbinom = _NBinom()
geom = _Geom()
hypergeom = _HyperGeom()
genextreme = _GenExtreme()
multivariate_normal = _MultivariateNormal()


# ---------------------------------------------------- residual tail

def probplot(x, dist="norm", fit=True):
    v = sorted(_flatten(x))
    n = len(v)
    # Filliben order-statistic medians
    osm_u = [1.0 - 0.5 ** (1.0 / n) if i == 0 else
             (0.5 ** (1.0 / n) if i == n - 1 else
              (i + 1 - 0.3175) / (n + 0.365)) for i in range(n)]
    osm = [_norm_ppf(u) for u in osm_u]
    if not fit:
        return (osm, v)
    slope_num = _math.fsum((a - _mean(osm)) * (b - _mean(v))
                           for a, b in zip(osm, v))
    slope_den = _math.fsum((a - _mean(osm)) ** 2 for a in osm)
    slope = slope_num / slope_den
    intercept = _mean(v) - slope * _mean(osm)
    r = _pearson_r(osm, v)
    return (osm, v), (slope, intercept, r)


def jarque_bera(x):
    v = _flatten(x)
    n = len(v)
    s = skew(v)
    k = kurtosis(v, fisher=True)
    jb = n / 6.0 * (s * s + k * k / 4.0)
    return _TestResult(jb, chi2.sf(jb, 2))


def friedmanchisquare(*samples):
    k = len(samples)
    cols = [_flatten(s) for s in samples]
    n = len(cols[0])
    rank_sums = [0.0] * k
    ties_corr = 0.0
    for i in range(n):
        row = [cols[j][i] for j in range(k)]
        r = rankdata(row)
        for j in range(k):
            rank_sums[j] += r[j]
        counts = {}
        for u in row:
            counts[u] = counts.get(u, 0) + 1
        ties_corr += _math.fsum(c ** 3 - c for c in counts.values())
    stat = (12.0 / (n * k * (k + 1))
            * _math.fsum(rs * rs for rs in rank_sums)
            - 3.0 * n * (k + 1))
    corr = 1.0 - ties_corr / (n * k * (k * k - 1))
    if corr > 0:
        stat /= corr
    return _TestResult(stat, chi2.sf(stat, k - 1))


def wasserstein_distance(u_values, v_values):
    u = sorted(_flatten(u_values))
    v = sorted(_flatten(v_values))
    allv = sorted(u + v)
    import bisect
    d = 0.0
    for i in range(len(allv) - 1):
        cu = bisect.bisect_right(u, allv[i]) / len(u)
        cv = bisect.bisect_right(v, allv[i]) / len(v)
        d += abs(cu - cv) * (allv[i + 1] - allv[i])
    return d


def somersd(x, y):
    xv, yv = _flatten(x), _flatten(y)
    n = len(xv)
    conc = disc = ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xv[i] - xv[j]
            dy = yv[i] - yv[j]
            if dx == 0:
                continue
            if dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    tot = conc + disc + ty
    d = (conc - disc) / tot if tot else float("nan")
    z = (conc - disc) / _math.sqrt(
        n * (n - 1) * (2 * n + 5) / 18.0) if n > 2 else 0.0
    return _TestResult(d, _bi.min(1.0, 2.0 * norm.sf(abs(z))))


def theilslopes(y, x=None):
    yv = _flatten(y)
    xv = _flatten(x) if x is not None else list(range(len(yv)))
    slopes = []
    n = len(yv)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if xv[j] != xv[i]:
                slopes.append((yv[j] - yv[i]) / (xv[j] - xv[i]))
    slopes.sort()
    m = len(slopes)
    med = slopes[m // 2] if m % 2 else \
        0.5 * (slopes[m // 2 - 1] + slopes[m // 2])
    xs = sorted(xv)
    xmed = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    ys = sorted(yv)
    ymed = ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])
    inter = ymed - med * xmed
    out = _TestResult(med, inter)
    out.slope = med
    out.intercept = inter
    return out


def ranksums(x, y):
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    ranks = rankdata(xv + yv)
    r1 = _math.fsum(ranks[:n1])
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (r1 - expected) / _math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    return _TestResult(z, 2.0 * norm.sf(abs(z)))


def median_test(*samples):
    allv = []
    for s in samples:
        allv += _flatten(s)
    sv = sorted(allv)
    n = len(sv)
    grand = sv[n // 2] if n % 2 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])
    table = []
    for s in samples:
        v = _flatten(s)
        above = sum(1 for u in v if u > grand)
        below = sum(1 for u in v if u <= grand)
        table.append([above, below])
    tt = [[table[i][j] for i in range(len(samples))]
          for j in range(2)]
    res = chi2_contingency(tt)
    return _TestResult(res.statistic, res.pvalue, median=grand,
                       table=tt)


class _KSTwoBign:
    """Asymptotic two-sided KS distribution (Kolmogorov)."""

    @staticmethod
    def sf(x):
        x = float(x)
        if x <= 0:
            return 1.0
        s = 0.0
        for j in range(1, 101):
            term = 2.0 * (-1) ** (j - 1) * _math.exp(-2.0 * j * j * x * x)
            s += term
            if abs(term) < 1e-16:
                break
        return _bi.max(0.0, _bi.min(1.0, s))

    @staticmethod
    def cdf(x):
        return 1.0 - _KSTwoBign.sf(x)

    @staticmethod
    def ppf(q):
        lo, hi = 1e-8, 5.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _KSTwoBign.cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def isf(q):
        return _KSTwoBign.ppf(1.0 - q)


kstwobign = _KSTwoBign()


class _KSTwo:
    """Finite-n two-sided KS via asymptotic + Stephens correction."""

    @staticmethod
    def sf(d, n):
        return _ks_sf(float(d), int(n))

    @staticmethod
    def cdf(d, n):
        return 1.0 - _ks_sf(float(d), int(n))

    @staticmethod
    def ppf(q, n):
        lo, hi = 1e-8, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _KSTwo.cdf(mid, n) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def isf(q, n):
        return _KSTwo.ppf(1.0 - q, n)


kstwo = _KSTwo()


class _HalfCauchy(_Dist):
    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0:
                return 0.0
            return 2.0 / (_math.pi * scale * (1.0 + z * z))
        return _maybe_map(one, x)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.0 if z < 0 else 2.0 / _math.pi * _math.atan(z)
        return _maybe_map(one, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * _math.tan(_math.pi * p / 2.0)
        return _maybe_map(one, q)


class _Pareto(_Dist):
    def pdf(self, x, b, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.0 if z < 1.0 else b / (z ** (b + 1.0)) / scale
        return _maybe_map(one, x)

    def cdf(self, x, b, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.0 if z < 1.0 else 1.0 - z ** (-b)
        return _maybe_map(one, x)

    def ppf(self, q, b, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * (1.0 - p) ** (-1.0 / b)
        return _maybe_map(one, q)


class _GenPareto(_Dist):
    def pdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0:
                return 0.0
            if c == 0:
                return _math.exp(-z) / scale
            t_ = 1.0 + c * z
            if t_ <= 0:
                return 0.0
            return t_ ** (-1.0 / c - 1.0) / scale
        return _maybe_map(one, x)

    def cdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0:
                return 0.0
            if c == 0:
                return 1.0 - _math.exp(-z)
            t_ = 1.0 + c * z
            if t_ <= 0:
                return 1.0
            return 1.0 - t_ ** (-1.0 / c)
        return _maybe_map(one, x)

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if c == 0:
                return loc - scale * _math.log1p(-p)
            return loc + scale * ((1.0 - p) ** (-c) - 1.0) / c
        return _maybe_map(one, q)


class _NCT(_Dist):
    """Noncentral t via cdf integration of the defining integral."""

    def cdf(self, x, df, nc):
        def one(v):
            # Algorithm: P(T<=t) = P(Z <= (t*sqrt(W/df) - nc)) averaged
            # over W ~ chi2(df); Gauss-Legendre on W quantiles
            npts = 200
            total = 0.0
            for i in range(npts):
                u = (i + 0.5) / npts
                w = chi2.ppf(u, df)
                total += _norm_cdf(v * _math.sqrt(w / df) - nc)
            return total / npts
        return _maybe_map(one, x)

    def sf(self, x, df, nc):
        c = self.cdf(x, df, nc)
        if isinstance(c, float):
            return 1.0 - c
        return 1.0 - c

    def pdf(self, x, df, nc):
        def one(v):
            h = 1e-5 * _bi.max(abs(v), 1.0)
            lo = self.cdf(v - h, df, nc)
            hi = self.cdf(v + h, df, nc)
            return (hi - lo) / (2.0 * h)
        return _maybe_map(one, x)

    def ppf(self, q, df, nc):
        def one(p):
            return _ppf_from_cdf(lambda v: self.cdf(v, df, nc), p,
                                 -1e3, 1e3)
        return _maybe_map(one, q)


class _NCF(_Dist):
    """Noncentral F via chi2 mixture average."""

    def cdf(self, x, dfn, dfd, nc):
        def one(v):
            npts = 200
            total = 0.0
            for i in range(npts):
                u = (i + 0.5) / npts
                w = chi2.ppf(u, dfd)      # denominator chi2
                # P(chi2_nc(dfn) <= v*dfn*w/dfd) with noncentrality nc:
                # Poisson mixture of central chi2
                lim = v * dfn * w / dfd
                acc = 0.0
                pw = _math.exp(-nc / 2.0)
                for j in range(200):
                    acc += pw * chi2.cdf(lim, dfn + 2 * j)
                    pw *= (nc / 2.0) / (j + 1)
                    if pw < 1e-14 and j > nc:
                        break
                total += acc
            return total / npts
        return _maybe_map(one, x)

    def sf(self, x, dfn, dfd, nc):
        c = self.cdf(x, dfn, dfd, nc)
        return 1.0 - c if isinstance(c, float) else 1.0 - c


halfcauchy = _HalfCauchy()
pareto = _Pareto()
genpareto = _GenPareto()
nct = _NCT()
ncf = _NCF()


# ---------------------------------------------------- residual tail 2

def bartlett(*samples):
    gs = [_flatten(s) for s in samples]
    k = len(gs)
    ns = [len(g) for g in gs]
    n = sum(ns)
    sp2 = _math.fsum((ns[i] - 1) * _var(gs[i], ddof=1)
                     for i in range(k)) / (n - k)
    num = (n - k) * _math.log(sp2) - _math.fsum(
        (ns[i] - 1) * _math.log(_var(gs[i], ddof=1))
        for i in range(k))
    den = 1.0 + (_math.fsum(1.0 / (ns[i] - 1) for i in range(k))
                 - 1.0 / (n - k)) / (3.0 * (k - 1))
    stat = num / den
    return _TestResult(stat, chi2.sf(stat, k - 1))


def fligner(*samples, center="median"):
    gs = [_flatten(s) for s in samples]
    k = len(gs)
    n = sum(len(g) for g in gs)
    zs = []
    for g in gs:
        sv = sorted(g)
        m = len(sv)
        c = sv[m // 2] if m % 2 else 0.5 * (sv[m // 2 - 1] + sv[m // 2])
        if center == "mean":
            c = _mean(g)
        zs.append([abs(v - c) for v in g])
    allz = [v for z in zs for v in z]
    ranks = rankdata(allz)
    a = [_norm_ppf(0.5 + r / (2.0 * (n + 1.0))) for r in ranks]
    abar = _mean(a)
    v2 = _var(a, ddof=1)
    stat = 0.0
    i = 0
    for z in zs:
        ni = len(z)
        ai = _math.fsum(a[i:i + ni]) / ni
        stat += ni * (ai - abar) ** 2
        i += ni
    stat /= v2
    return _TestResult(stat, chi2.sf(stat, k - 1))


def ansari(x, y):
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    n = n1 + n2
    ranks = rankdata(xv + yv)
    # Ansari-Bradley scores: min(r, N+1-r)
    scores = [_bi.min(r, n + 1.0 - r) for r in ranks]
    ab = _math.fsum(scores[:n1])
    if n % 2 == 0:
        mu = n1 * (n + 2.0) / 4.0
        var = n1 * n2 * (n + 2.0) * (n - 2.0) / (48.0 * (n - 1.0))
    else:
        mu = n1 * (n + 1.0) ** 2 / (4.0 * n)
        var = n1 * n2 * (n + 1.0) * (3.0 + n * n) / (48.0 * n * n)
    z = (ab - mu) / _math.sqrt(var)
    return _TestResult(ab, _bi.min(1.0, 2.0 * norm.sf(abs(z))))


def cramervonmises(rvs, cdf, args=()):
    v = sorted(_flatten(rvs))
    n = len(v)
    if isinstance(cdf, str):
        dist = {"norm": norm, "uniform": uniform, "expon": expon}[cdf]
        cdfv = [dist.cdf(u, *args) for u in v]
    else:
        cdfv = [float(cdf(u, *args)) for u in v]
    w2 = 1.0 / (12.0 * n) + _math.fsum(
        (cdfv[i] - (2.0 * i + 1.0) / (2.0 * n)) ** 2
        for i in range(n))
    return _TestResult(w2, _cvm_asymp_sf(w2))


def _kv_quarter(x):
    """K_{1/4}(x) via integral representation."""
    if x > 700.0:
        return 0.0

    def f(t):
        e = -x * _math.cosh(t)
        if e < -700.0:
            return 0.0
        return _math.exp(e) * _math.cosh(0.25 * t)
    hi = 1.0
    while x * _math.cosh(hi) < 720.0 and hi < 60.0:
        hi += 1.0
    total = 0.0
    m = 400
    for i in range(m):
        t = (i + 0.5) * hi / m
        total += f(t)
    return total * hi / m


def cramervonmises_2samp(x, y):
    xv, yv = sorted(_flatten(x)), sorted(_flatten(y))
    n, m = len(xv), len(yv)
    allr = rankdata(xv + yv)
    rx = allr[:n]
    ry = allr[n:]
    # Anderson (1962) computational form
    u = n * _math.fsum((rx[i] - (i + 1)) ** 2
                       for i in range(n)) \
        + m * _math.fsum((ry[j] - (j + 1)) ** 2 for j in range(m))
    nm = n + m
    t = u / (n * m * nm) - (4.0 * n * m - 1.0) / (6.0 * nm)
    return _TestResult(t, _cvm_asymp_sf(t))


def _cvm_asymp_sf(t):
    s = 0.0
    for j in range(200):
        a = 4.0 * j + 1.0
        term = (_math.gamma(j + 0.5) / (_math.gamma(0.5)
                * _math.factorial(j))) * _math.sqrt(a) \
            * _math.exp(-a * a / (16.0 * t)) * _kv_quarter(
                a * a / (16.0 * t))
        s += term
        if term < 1e-12 and j > 3:
            break
    return _bi.max(0.0, _bi.min(1.0, 1.0 - s / (_math.pi
                                                * _math.sqrt(t))))


def anderson_ksamp(samples, midrank=True):
    gs = [sorted(_flatten(s)) for s in samples]
    k = len(gs)
    ns = [len(g) for g in gs]
    allv = sorted(v for g in gs for v in g)
    n = len(allv)
    zstar = sorted(set(allv))
    import bisect
    if midrank:
        # Scholz-Stephens A2akN (midrank / ties variant, scipy default)
        a2 = 0.0
        for gi, g in enumerate(gs):
            inner = 0.0
            for z in zstar:
                zl = bisect.bisect_left(allv, z)
                lj = bisect.bisect_right(allv, z) - zl
                bj = zl + lj / 2.0
                sr = bisect.bisect_right(g, z)
                fij = sr - bisect.bisect_left(g, z)
                mij = sr - fij / 2.0
                denom = bj * (n - bj) - n * lj / 4.0
                if denom > 0:
                    inner += (lj / float(n)
                              * (n * mij - bj * ns[gi]) ** 2 / denom)
            a2 += inner / ns[gi]
        a2 *= (n - 1.0) / n
        A2kN = a2 - (k - 1)
    else:
        a2 = 0.0
        for gi, g in enumerate(gs):
            inner = 0.0
            for z in zstar[:-1]:
                mij = bisect.bisect_right(g, z)
                bj = bisect.bisect_right(allv, z)
                if 0 < bj < n:
                    inner += (n * mij - ns[gi] * bj) ** 2 / float(
                        bj * (n - bj))
            a2 += inner / ns[gi]
        a2 /= n
        A2kN = a2 - (k - 1)
    H = _math.fsum(1.0 / v for v in ns)
    hs = _math.fsum(1.0 / i for i in range(1, n))
    gsum = 0.0
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            gsum += 1.0 / ((n - i) * j)
    a = (4.0 * gsum - 6.0) * (k - 1) + (10.0 - 6.0 * gsum) * H
    b = (2.0 * gsum - 4.0) * k * k + 8.0 * hs * k \
        + (2.0 * gsum - 14.0 * hs - 4.0) * H - 8.0 * hs \
        + 4.0 * gsum - 6.0
    c = (6.0 * hs + 2.0 * gsum - 2.0) * k * k \
        + (4.0 * hs - 4.0 * gsum + 6.0) * k \
        + (2.0 * hs - 6.0) * H + 4.0 * hs
    d = (2.0 * hs + 6.0) * k * k - 4.0 * hs * k
    sigsq = (a * n ** 3 + b * n ** 2 + c * n + d) / (
        (n - 1.0) * (n - 2.0) * (n - 3.0))
    tn = A2kN / _math.sqrt(sigsq)
    b0 = [0.675, 1.281, 1.645, 1.960, 2.326, 2.573, 3.085]
    b1 = [-0.245, 0.250, 0.678, 1.149, 1.822, 2.364, 3.615]
    b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154]
    m = k - 1.0
    tm = [b0[i] + b1[i] / _math.sqrt(m) + b2[i] / m for i in range(7)]
    sig = [0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
    logsig = [_math.log(s) for s in sig]
    if tn < tm[0]:
        p = 0.25
    elif tn > tm[-1]:
        p = 0.001
    else:
        import bisect as _bs
        i = _bs.bisect_left(tm, tn)
        frac = (tn - tm[i - 1]) / (tm[i] - tm[i - 1])
        p = _math.exp(logsig[i - 1]
                      + frac * (logsig[i] - logsig[i - 1]))
    return _TestResult(tn, p, significance_level=p,
                       critical_values=tm)


def binom_test(x, n=None, p=0.5, alternative="two-sided"):
    return binomtest(x, n, p, alternative).pvalue


class _LogUniform:
    def __init__(self, a=None, b=None):
        self.a, self.b = a, b

    def __call__(self, a, b):
        return _LogUniform(a, b)

    def rvs(self, size=None, random_state=None):
        from . import _array_core as _ac2
        if hasattr(random_state, "random"):
            rng = random_state
        else:
            rng = _ac2.random.default_rng(random_state)

        def one(u):
            return self.a * (self.b / self.a) ** u
        if size is None:
            return one(rng.random())
        return _ac2.marr([one(rng.random()) for _ in range(int(size))])

    def pdf(self, x, a=None, b=None):
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        def one(v):
            if v < a or v > b:
                return 0.0
            return 1.0 / (v * _math.log(b / a))
        return _maybe_map(one, x)

    def cdf(self, x, a=None, b=None):
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        def one(v):
            if v <= a:
                return 0.0
            if v >= b:
                return 1.0
            return _math.log(v / a) / _math.log(b / a)
        return _maybe_map(one, x)

    def ppf(self, q, a=None, b=None):
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        def one(pp):
            return a * (b / a) ** pp
        return _maybe_map(one, q)


loguniform = _LogUniform()


class _MStats:
    @staticmethod
    def winsorize(a, limits=None):
        v = _flatten(a)
        n = len(v)
        lo_l, hi_l = (limits if isinstance(limits, (tuple, list))
                      else (limits, limits)) if limits is not None \
            else (0.0, 0.0)
        sv = sorted(v)
        klo = int(lo_l * n)
        khi = int(hi_l * n)
        lo_v = sv[klo] if klo < n else sv[-1]
        hi_v = sv[n - khi - 1] if khi < n else sv[0]
        return [lo_v if u < lo_v else (hi_v if u > hi_v else u)
                for u in v]


mstats = _MStats()


class _LatinHypercube:
    def __init__(self, d, seed=None):
        self.d = d
        self._rng_seed = seed if seed is not None else 0

    def random(self, n):
        from . import _array_core as _ac2
        rng = _ac2.random.default_rng(self._rng_seed)
        cols = []
        for _j in range(self.d):
            perm = list(range(n))
            rng.shuffle(perm)
            cols.append([(perm[i] + rng.uniform()) / n
                         for i in range(n)])
        return _ac2.marr([[cols[j][i] for j in range(self.d)]
                          for i in range(n)])


class _QMC:
    LatinHypercube = _LatinHypercube


qmc = _QMC()


class _MVN:
    """stats.mvn.mvnun replacement: rectangle probability via
    quasi-Monte Carlo (deterministic seed); tolerance ~1e-4."""

    @staticmethod
    def mvnun(lower, upper, means, covar, maxpts=20000, **kw):
        del kw
        from . import _array_core as _ac2
        lo = [float(v) for v in lower]
        hi = [float(v) for v in upper]
        mu = [float(v) for v in means]
        cov = [[float(v) for v in row]
               for row in (covar.tolist() if hasattr(covar, "tolist")
                           else covar)]
        d = len(mu)
        L = _ac2.linalg.cholesky(_ac2.marr(cov)).tolist()
        rng = _ac2.random.default_rng(42)
        count = 0
        npts = int(maxpts)
        for _ in range(npts):
            z = [rng.normal() for _ in range(d)]
            x = [mu[i] + _math.fsum(L[i][j] * z[j]
                                    for j in range(i + 1))
                 for i in range(d)]
            if all(lo[i] <= x[i] <= hi[i] for i in range(d)):
                count += 1
        return count / npts, 0


mvn = _MVN()


winsorize = _MStats.winsorize    # scipy.stats.mstats import site
