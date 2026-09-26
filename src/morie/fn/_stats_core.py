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
from functools import lru_cache as _lru_cache


def _erf(x):
    return _math.erf(x)


def _norm_cdf(z):
    # erfc keeps full relative accuracy in the lower tail, where
    # 1 + erf(z / sqrt 2) cancels to 0 below z of about -8
    return 0.5 * _math.erfc(-z / _math.sqrt(2.0))


def _log_ndtr(z):
    """log Phi(z).  Direct where Phi(z) is representable with full
    relative accuracy; below z = -20 the asymptotic series
    log Phi(z) = -z^2/2 - log(-z) - log(2 pi)/2
                 + log(1 - 1/z^2 + 3/z^4 - 15/z^6 + 105/z^8)
    (Abramowitz and Stegun 26.2.12), which stays finite long after
    Phi(z) underflows."""
    if z > -20.0:
        c = 0.5 * _math.erfc(-z / _math.sqrt(2.0))
        return _math.log(c) if z < 5.0 else _math.log1p(-0.5 * _math.erfc(z / _math.sqrt(2.0)))
    z2 = z * z
    # past |z| = 1e8 the correction is below 1e-16 (and z2**5 would overflow)
    series = 1.0 if z2 > 1e16 else \
        1.0 - 1.0 / z2 + 3.0 / z2 ** 2 - 15.0 / z2 ** 3 + 105.0 / z2 ** 4 - 945.0 / z2 ** 5
    return -0.5 * z2 - _math.log(-z) - 0.5 * _math.log(2.0 * _math.pi) + _math.log(series)


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


def _gammainc_q(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), computed
    directly (Lentz continued fraction) where it is small, so the upper
    tail keeps full relative accuracy instead of cancelling in 1 - P."""
    if x < 0 or a <= 0:
        raise ValueError("invalid arguments")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gammainc_p(a, x)
    return _math.exp(_log_gammainc_q_cf(a, x))


def _log_gammainc_q_cf(a, x):
    """log Q(a, x) for x >= a + 1 by the Lentz continued fraction, in log
    space: Q itself underflows long before its log is out of range."""
    ln_pre = a * _math.log(x) - x - _math.lgamma(a)
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
    return ln_pre + _math.log(h)


def _log_gammainc_q(a, x):
    """log of the regularized upper incomplete gamma."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return _math.log1p(-_gammainc_p(a, x))
    return _log_gammainc_q_cf(a, x)


def _stirling_tail(z):
    """ln Gamma(z) - [(z - 1/2) ln z - z + ln(2 pi)/2]: the Stirling
    series, accurate to double precision for z >= 20."""
    z2 = z * z
    return (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - (1.0 / 1680.0
            - (1.0 / 1188.0 - 691.0 / 360360.0 / z2) / z2) / z2) / z2) / z2) / z


def _lbeta(a, b):
    """ln B(a, b) without the cancellation of lgamma(a) + lgamma(b) -
    lgamma(a + b) when one argument is large: for the larger argument
    b >= 20, ln Gamma(b) - ln Gamma(a + b) is formed from the Stirling
    series as -a ln b - (a + b - 1/2) log1p(a/b) + a + S(b) - S(a + b),
    so t, F and beta probabilities stay accurate at huge degrees of
    freedom (the plain difference lost ~df * eps)."""
    if a > b:
        a, b = b, a
    if b < 20.0:
        return _math.lgamma(a) + _math.lgamma(b) - _math.lgamma(a + b)
    if a >= 20.0:
        # both large: Stirling on all three terms
        return (0.5 * _math.log(2.0 * _math.pi) + (a - 0.5) * _math.log(a)
                + (b - 0.5) * _math.log(b) - (a + b - 0.5) * _math.log(a + b)
                + _stirling_tail(a) + _stirling_tail(b) - _stirling_tail(a + b))
    diff = (-a * _math.log(b) - (a + b - 0.5) * _math.log1p(a / b) + a
            + _stirling_tail(b) - _stirling_tail(a + b))
    return _math.lgamma(a) + diff


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), Lentz continued fraction."""
    if not 0.0 <= x <= 1.0 or a <= 0 or b <= 0:
        raise ValueError("invalid arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_pre = (-_lbeta(a, b)
              + a * _math.log(x) + b * _math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return _math.exp(ln_pre) * _betacf(a, b, x) / a
    return 1.0 - _math.exp(ln_pre) * _betacf(b, a, 1.0 - x) / b


def _betaincc(a, b, x):
    """1 - I_x(a, b) = I_{1-x}(b, a), computed directly on the side where
    it is small so an upper tail does not cancel."""
    if not 0.0 <= x <= 1.0 or a <= 0 or b <= 0:
        raise ValueError("invalid arguments")
    if x == 0.0:
        return 1.0
    if x == 1.0:
        return 0.0
    ln_pre = (-_lbeta(a, b) + a * _math.log(x) + b * _math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return 1.0 - _math.exp(ln_pre) * _betacf(a, b, x) / a
    return _math.exp(ln_pre) * _betacf(b, a, 1.0 - x) / b


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


def _invert(cdf, lo, hi, p=None, q=None, sf=None, iters=3000):
    """Solve cdf(x) = p, or sf(x) = q, for a monotone distribution.

    The tail that carries the information is used: a lower probability
    solves on the cdf, an upper one on the sf (given directly as q, or as
    1 - p when p > 1/2 and an sf is available), so neither tail loses
    digits to 1 - p.  Bisection is geometric when the bracket has one
    sign -- quantiles like 1e-200 need relative, not absolute, steps --
    and stops at relative width 2e-16.
    """
    if q is not None and q > 0.5:
        # an upper probability above 1/2 is a lower one below it: 1 - q is
        # exact in floating point, the sf near 1 is not
        use_sf, target = False, 1.0 - q
    elif q is not None:
        use_sf, target = True, q
    elif sf is not None and p > 0.5:
        use_sf, target = True, 1.0 - p
    else:
        use_sf, target = False, p

    def h(x):
        return target - sf(x) if use_sf else cdf(x) - target

    grow = 0
    while h(hi) < 0 and grow < 2000:
        hi = hi * 2.0 if hi > 0 else (hi / 2.0 if hi < 0 else 1.0)
        grow += 1
    while h(lo) > 0 and grow < 4000:
        lo = lo * 2.0 if lo < 0 else (lo / 2.0 if lo > 0 else -1.0)
        grow += 1
    for _ in range(iters):
        if lo == 0.0 and hi > 0.0:
            mid = hi * 1e-10 if hi * 1e-10 > 1e-300 else hi * 0.5
        elif hi == 0.0 and lo < 0.0:
            mid = lo * 1e-10 if -lo * 1e-10 > 1e-300 else lo * 0.5
        elif lo > 0.0 and hi > 2.0 * lo:
            mid = _math.sqrt(lo) * _math.sqrt(hi)
        elif hi < 0.0 and lo < 2.0 * hi:
            mid = -_math.sqrt(-lo) * _math.sqrt(-hi)
        else:
            mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        v = h(mid)
        if v < 0:
            lo = mid
        elif v > 0:
            hi = mid
        else:
            return mid
        if hi - lo <= 2e-16 * max(_bi_abs(lo), _bi_abs(hi)):
            break
    return 0.5 * (lo + hi)


def _ppf_from_cdf(cdf, p, lo, hi, iters=200, sf=None):
    """Monotone-cdf inversion (see _invert); p = 0 / 1 give the bracket."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return lo
        if p == 1.0:
            return hi
        raise ValueError("p must be in [0, 1]")
    return _invert(cdf, lo, hi, p=p, sf=sf)


def _bi_abs(v):
    return v if v >= 0 else -v


def _is_arraylike(v):
    return v is not None and not isinstance(v, (int, float, bool)) and (
        hasattr(v, "tolist") or isinstance(v, (list, tuple)))


def _bcast(one, *args):
    """Evaluate a scalar function elementwise over broadcast arguments,
    returning a marr of the broadcast shape (scipy's convention for
    array-valued distribution parameters)."""
    from . import _array_core as _ac2
    arrs = _ac2.broadcast_arrays(*[_ac2.asarray(a) if _is_arraylike(a)
                                   else _ac2.asarray([float(a)]) for a in args])
    shape = arrs[0].shape
    flat = [a.ravel().tolist() for a in arrs]
    vals = [one(*[float(f[i]) for f in flat]) for i in range(len(flat[0]))]
    return _ac2.marr(vals).reshape(shape) if len(shape) > 1 else _ac2.marr(vals)


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


def _rng_from(random_state):
    """A generator from a seed, an existing generator, or None."""
    from . import _array_core as _ac
    if hasattr(random_state, "random") and callable(random_state.random):
        return random_state
    return _ac.random.default_rng(random_state)


def _scalar(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, (list, tuple)):
        v = v[0]
    return float(v)


def _edge_wrap(name, fn):
    """scipy's answers at the edges, for every distribution at once.

    Applied to each subclass's cdf/sf/pdf/pmf/ppf/isf/logpdf/logcdf/logsf
    by _Dist.__init_subclass__: NaN in is NaN out; ppf/isf outside [0, 1]
    is NaN and at 0/1 is the support bound (discrete: lower - 1 at 0);
    cdf/sf/pdf/logpdf at +-inf are their limits; a discrete pmf off the
    integer support is 0 and a discrete cdf floors its argument; a
    domain error from the body (x outside the support) is pdf 0, logpdf
    -inf, cdf/sf at the nearer bound. Before this, ppf(1.0) returned the
    bisection cap (chi2: 13.0, t: 55.0), logpdf outside the support
    raised, and geom.cdf(-1) was -0.43.
    """
    import inspect as _inspect
    _params = list(_inspect.signature(fn).parameters)[2:]   # after self, x
    _takes_loc = "loc" in _params
    _takes_scale = "scale" in _params
    _n_pos = len(_params)

    def wrapped(self, x, *args, **kw):
        # scipy broadcasts x against array-valued shape, loc and scale
        # parameters (one rate per observation, say); the scalar path
        # below read only the first element of such a parameter
        if any(_is_arraylike(a) for a in args) or \
                any(_is_arraylike(v) for v in kw.values()):
            keys = list(kw)
            if all(_is_arraylike(v) or isinstance(v, (int, float)) for v in
                   list(args) + [kw[k] for k in keys]):
                na = len(args)

                def call(xv, *vals):
                    return wrapped(self, xv, *vals[:na],
                                   **dict(zip(keys, vals[na:])))
                return _bcast(call, x, *args, *[kw[k] for k in keys])
        # scipy's loc / scale on a body that has none: shift and scale
        # the argument (cdf/sf/pdf family) or the result (ppf/isf). Given
        # positionally past the body's own parameters, they are loc then
        # scale, as in t.ppf(q, df, loc, scale).
        loc = kw.pop("loc", 0.0) if not _takes_loc else None
        scale = kw.pop("scale", 1.0) if not _takes_scale else None
        if len(args) > _n_pos:
            extra, args = args[_n_pos:], args[:_n_pos]
            if loc is not None:
                loc = extra[0]
            if scale is not None and len(extra) > 1:
                scale = extra[1]
        loc = 0.0 if loc is None else float(loc)
        scale = 1.0 if scale is None else float(scale)
        affine = loc != 0.0 or scale != 1.0
        if affine and scale <= 0:
            return _math.nan
        if affine:
            if name in ("ppf", "isf"):
                base = wrapped(self, x, *args, **kw)
                return _maybe_map(lambda r: loc + scale * r, base)
            shifted = _maybe_map(lambda v: (v - loc) / scale, x)
            base = wrapped(self, shifted, *args, **kw)
            if name == "pdf":
                return _maybe_map(lambda r: r / scale, base)
            if name == "logpdf":
                return _maybe_map(lambda r: r - _math.log(scale), base)
            return base

        def one(v):
            v = float(v)
            if v != v:
                return _math.nan
            if name in ("ppf", "isf"):
                if v < 0.0 or v > 1.0:
                    return _math.nan
                if v in (0.0, 1.0):
                    lo, hi = self._bounds(*args, **kw)
                    at_lo = (v == 0.0) == (name == "ppf")
                    if at_lo:
                        return lo - 1.0 if self._discrete else lo
                    return hi
            elif v in (_math.inf, -_math.inf):
                up = v > 0
                if name == "cdf":
                    return 1.0 if up else 0.0
                if name == "sf":
                    return 0.0 if up else 1.0
                if name in ("pdf", "pmf"):
                    return 0.0
                if name == "logpdf":
                    return -_math.inf
                if name == "logcdf":
                    return 0.0 if up else -_math.inf
                if name == "logsf":
                    return -_math.inf if up else 0.0
            elif self._discrete and name in ("cdf", "sf", "pmf", "logcdf", "logsf"):
                lo, hi = self._bounds(*args, **kw)
                if name == "pmf":
                    if v != _math.floor(v) or v < lo or v > hi:
                        return 0.0
                else:
                    if v < lo:
                        return {"cdf": 0.0, "sf": 1.0, "logcdf": -_math.inf, "logsf": 0.0}[name]
                    v = float(_math.floor(v))
            try:
                return _scalar(fn(self, v, *args, **kw))
            except (ValueError, OverflowError, ZeroDivisionError):
                lo, hi = self._bounds(*args, **kw)
                if name in ("pdf", "pmf"):
                    return 0.0
                if name == "logpdf":
                    return -_math.inf
                if name == "cdf":
                    return 0.0 if v < lo else 1.0
                if name == "sf":
                    return 1.0 if v < lo else 0.0
                if name == "logcdf":
                    return -_math.inf if v < lo else 0.0
                if name == "logsf":
                    return 0.0 if v < lo else -_math.inf
                return _math.nan
        return _maybe_map(one, x)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    wrapped._edge_wrapped = True
    return wrapped


class _Frozen:
    """A distribution with its parameters bound (scipy's frozen form) for
    the classes whose methods take the parameters positionally."""

    _PARAMETRIC = ("mean", "var", "std", "median", "entropy", "support",
                   "moment", "stats", "interval", "expect")

    def __init__(self, dist, args, kw):
        self._dist, self._args, self._kw = dist, tuple(args), dict(kw)

    def __repr__(self):
        return f"{type(self._dist).__name__[1:].lower()}{self._args!r} frozen"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fn = getattr(self._dist, name)
        if not callable(fn):
            return fn
        args, kw = self._args, self._kw
        if name == "rvs":
            def rvs(size=None, random_state=None):
                return fn(*args, size=size, random_state=random_state, **kw)
            return rvs
        if name in self._PARAMETRIC:
            def parametric(*a, **k):        # moment(n), interval(alpha)
                return fn(*a, *args, **kw, **k)
            return parametric

        def at(x, *a, **k):                  # pdf(x), cdf(x), ppf(q), ...
            return fn(x, *args, *a, **kw, **k)
        return at


def _adaptive_simpson(fx, a, b, tol, depth):
    """Adaptive Simpson quadrature of fx on [a, b]."""
    c = 0.5 * (a + b)
    fa, fb, fc = fx(a), fx(b), fx(c)
    whole = (b - a) / 6.0 * (fa + 4.0 * fc + fb)

    def rec(a, b, fa, fb, fc, whole, tol, depth):
        c = 0.5 * (a + b)
        d, e = 0.5 * (a + c), 0.5 * (c + b)
        fd, fe = fx(d), fx(e)
        left = (c - a) / 6.0 * (fa + 4.0 * fd + fc)
        right = (b - c) / 6.0 * (fc + 4.0 * fe + fb)
        if depth <= 0 or _bi.abs(left + right - whole) <= 15.0 * tol:
            return left + right + (left + right - whole) / 15.0
        return (rec(a, c, fa, fc, fd, left, tol / 2.0, depth - 1)
                + rec(c, b, fc, fb, fe, right, tol / 2.0, depth - 1))
    return rec(a, b, fa, fb, fc, whole, tol, depth)


class _Dist:
    """Common frozen/unfrozen scipy-like surface."""

    def fit(self, data, *args, **kw):
        """Maximum-likelihood parameters. Closed forms exist for norm,
        expon, uniform, laplace, poisson, geom; the others raise."""
        raise NotImplementedError(
            f"{type(self).__name__[1:].lower()}.fit() is not available in the "
            "native core; use norm/expon/uniform/laplace, or fit by hand")

    _support = (-_math.inf, _math.inf)
    _discrete = False

    def _bounds(self, *args, **kw):
        """(lower, upper) of the support for these parameters."""
        return self._support

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for name in ("cdf", "sf", "pdf", "pmf", "ppf", "isf", "logpdf", "logcdf", "logsf"):
            fn = cls.__dict__.get(name)
            if callable(fn) and not getattr(fn, "_edge_wrapped", False):
                setattr(cls, name, _edge_wrap(name, fn))

    def __call__(self, *args, **kw):
        if type(self).__init__ is not object.__init__:
            return self.__class__(*args, **kw)     # stores loc/scale itself
        # the classes whose methods take the parameters positionally get
        # scipy's frozen form through a binder, so st.laplace(0, 2).rvs(3)
        # works like st.norm(0, 2).rvs(3)
        return _Frozen(self, args, kw)

    def rvs(self, *args, size=None, random_state=None, **kw):
        """Random variates by inverse transform: ppf applied to uniforms.

        Every distribution with a ppf gets rvs from this one definition
        (27 of 29 had none, so mrm_clt_demo() could not run). A frozen
        instance draws with its own parameters; an unfrozen one takes
        them as the positional/keyword arguments its ppf takes.
        ``random_state`` may be a seed or an existing default_rng()
        generator, as in numpy.
        """
        from . import _array_core as _ac
        rng = _rng_from(random_state)
        if size is None:
            return float(self.ppf(rng.random(), *args, **kw))
        if isinstance(size, (tuple, list)):
            dims = [int(d) for d in size]
            n = 1
            for d in dims:
                n *= d
        else:
            dims, n = [int(size)], int(size)
        out = [float(self.ppf(rng.random(), *args, **kw)) for _ in range(n)]
        if len(dims) == 2:
            r, c = dims
            return _ac.marr([out[i * c:(i + 1) * c] for i in range(r)])
        return _ac.marr(out)

    def sf(self, x, *args, **kw):
        c = self.cdf(x, *args, **kw)
        if isinstance(c, list):
            return [1.0 - v for v in c]
        return 1.0 - c

    def isf(self, q, *args, **kw):
        """Upper quantile.  When the distribution has its own sf the
        equation sf(x) = q is solved directly, so a tiny q keeps its
        digits (ppf(1 - q) cannot: 1 - 1e-20 is 1.0)."""
        has_sf = type(self).sf is not _Dist.sf

        def one(v):
            v = float(v)
            if v != v or not 0.0 <= v <= 1.0:
                return _math.nan
            if not has_sf or v >= 0.5 or v in (0.0, 1.0):
                return self.ppf(1.0 - v, *args, **kw)
            try:
                lo_b, hi_b = self._bounds(*args, **kw)
            except TypeError:
                lo_b, hi_b = self._bounds()
            med = float(self.ppf(0.5, *args, **kw))
            lo = med
            hi = hi_b if hi_b != _math.inf else (_bi_abs(med) * 2.0 + 1.0)
            return _invert(lambda x: float(self.cdf(x, *args, **kw)), lo, hi, q=v,
                           sf=lambda x: float(self.sf(x, *args, **kw)))
        return _maybe_map(one, q)

    def logpdf(self, x, *args, **kw):
        def one(v):
            p = _scalar(self.pdf(v, *args, **kw))
            if p != p:
                return _math.nan
            return _math.log(p) if p > 0 else -_math.inf
        return _maybe_map(one, x)
    def logcdf(self, x, *a, **k):
        # above the median log(cdf) is log(1 - sf): log1p(-sf) keeps the
        # digits that log(0.99999...) throws away
        has_sf = type(self).sf is not _Dist.sf

        def one(v):
            c = _scalar(self.cdf(v, *a, **k))
            if c != c:
                return _math.nan
            if c > 0.5 and has_sf:
                return _math.log1p(-_scalar(self.sf(v, *a, **k)))
            return _math.log(c) if c > 0 else -_math.inf
        return _maybe_map(one, x)

    # -- the moment interface scipy gives every distribution: mean, var,
    #    std, median, entropy, moment, interval, stats, support, expect.
    #    Closed forms in subclasses take precedence; this default
    #    integrates the density (or sums the mass) numerically.
    def _bounds_for(self, *args, **kw):
        try:
            b = self._bounds(*args, **kw)
        except TypeError:
            b = self._support
        return float(b[0]), float(b[1])

    def _quantile_range(self, *args, **kw):
        lo, hi = self._bounds_for(*args, **kw)
        # 1e-16 tails: cutting at 1e-10 dropped ~2e-10 of the mass, a
        # 1e-9 relative error in every numerically integrated moment
        qlo = _scalar(self.ppf(1e-16, *args, **kw))
        qhi = _scalar(self.isf(1e-16, *args, **kw))
        if lo == -_math.inf or qlo > lo:
            lo = qlo
        if hi == _math.inf or qhi < hi:
            hi = qhi
        return lo, hi

    def expect(self, func=None, *args, **kw):
        """E[func(X)] (func defaults to the identity)."""
        g = func if func is not None else (lambda v: v)
        if self._discrete:
            lo, hi = self._bounds_for(*args, **kw)
            k = int(lo) if lo > -_math.inf else int(_scalar(self.ppf(1e-14, *args, **kw)))
            kmax = int(hi) if hi < _math.inf else k + 10 ** 6
            total, mass = 0.0, 0.0
            while k <= kmax:
                p = _scalar(self.pmf(k, *args, **kw))
                total += p * g(k)
                mass += p
                if mass > 1.0 - 1e-14 and hi == _math.inf and k > lo + 5:
                    break
                k += 1
            return total
        lo, hi = self._quantile_range(*args, **kw)

        def fx(v):
            p = _scalar(self.pdf(v, *args, **kw))
            return 0.0 if p != p else p * g(v)
        # knots at quantiles so a peaked or heavy-tailed density gets its
        # resolution where the mass is; adaptive Simpson on each piece
        knots = [lo]
        for q in (1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999, 1.0 - 1e-6,
                  1.0 - 1e-9):
            v = _scalar(self.ppf(q, *args, **kw))
            if v == v and knots[-1] < v < hi:
                knots.append(v)
        knots.append(hi)
        total = 0.0
        for a, b in zip(knots[:-1], knots[1:]):
            total += _adaptive_simpson(fx, a, b, 1e-11, 30)
        return total

    def mean(self, *args, **kw):
        return self.expect(None, *args, **kw)

    def var(self, *args, **kw):
        mu = self.mean(*args, **kw)
        return self.expect(lambda v: (v - mu) ** 2, *args, **kw)

    def std(self, *args, **kw):
        return _math.sqrt(self.var(*args, **kw))

    def median(self, *args, **kw):
        return _scalar(self.ppf(0.5, *args, **kw))

    def moment(self, order, *args, **kw):
        """Raw moment E[X**order]."""
        return self.expect(lambda v: v ** order, *args, **kw)

    def entropy(self, *args, **kw):
        if self._discrete:
            return self.expect(lambda k: -_math.log(_scalar(self.pmf(k, *args, **kw)))
                               if _scalar(self.pmf(k, *args, **kw)) > 0 else 0.0,
                               *args, **kw)
        return self.expect(lambda v: -_math.log(_scalar(self.pdf(v, *args, **kw)))
                           if _scalar(self.pdf(v, *args, **kw)) > 0 else 0.0,
                           *args, **kw)

    def interval(self, confidence, *args, **kw):
        """Equal-tailed interval containing ``confidence`` of the mass."""
        c = float(confidence)
        if not 0.0 <= c <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return (_scalar(self.ppf((1.0 - c) / 2.0, *args, **kw)),
                _scalar(self.ppf((1.0 + c) / 2.0, *args, **kw)))

    def support(self, *args, **kw):
        return self._bounds_for(*args, **kw)

    def stats(self, *args, moments="mv", **kw):
        out = []
        for ch in moments:
            if ch == "m":
                out.append(self.mean(*args, **kw))
            elif ch == "v":
                out.append(self.var(*args, **kw))
            elif ch == "s":
                mu, sd = self.mean(*args, **kw), self.std(*args, **kw)
                out.append(self.expect(lambda v, mu=mu, sd=sd: ((v - mu) / sd) ** 3,
                                       *args, **kw))
            elif ch == "k":
                mu, sd = self.mean(*args, **kw), self.std(*args, **kw)
                out.append(self.expect(lambda v, mu=mu, sd=sd: ((v - mu) / sd) ** 4,
                                       *args, **kw) - 3.0)
            else:
                raise ValueError("moments must be a combination of m, v, s, k")
        return tuple(out)

    def logsf(self, x, *a, **k):
        # below the median log(sf) is log(1 - cdf): use log1p(-cdf)
        def one(v):
            sv = _scalar(self.sf(v, *a, **k))
            if sv != sv:
                return _math.nan
            if sv > 0.5:
                return _math.log1p(-_scalar(self.cdf(v, *a, **k)))
            return _math.log(sv) if sv > 0 else -_math.inf
        return _maybe_map(one, x)


class _Norm(_Dist):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def _ls(self, loc, scale):
        return (self.loc if loc is None else float(loc),
                self.scale if scale is None else float(scale))

    def mean(self, loc=None, scale=None):
        return self._ls(loc, scale)[0]

    def var(self, loc=None, scale=None):
        """Variance scale**2."""
        s = self._ls(loc, scale)[1]
        return s * s

    def std(self, loc=None, scale=None):
        return self._ls(loc, scale)[1]

    def median(self, loc=None, scale=None):
        return self._ls(loc, scale)[0]

    def entropy(self, loc=None, scale=None):
        s = self._ls(loc, scale)[1]
        return 0.5 * _math.log(2.0 * _math.pi * _math.e * s * s)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def pdf(self, x, loc=None, scale=None):
        # either argument alone overrides the frozen value (scipy's
        # norm.cdf(x, scale=s) keeps loc = 0); the old test on loc only
        # silently dropped a scale passed without a loc
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: _norm_pdf(d._z(v)) / d.scale, x)

    def cdf(self, x, loc=None, scale=None):
        # either argument alone overrides the frozen value (scipy's
        # norm.cdf(x, scale=s) keeps loc = 0); the old test on loc only
        # silently dropped a scale passed without a loc
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: _norm_cdf(d._z(v)), x)

    def sf(self, x, loc=None, scale=None):
        # 0.5 erfc(z / sqrt 2) directly: 1 - cdf cancels in the upper tail
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: _norm_cdf(-d._z(v)), x)

    def logpdf(self, x, loc=None, scale=None):
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: -0.5 * d._z(v) ** 2 - 0.5 * _math.log(2.0 * _math.pi)
                          - _math.log(d.scale), x)

    def logcdf(self, x, loc=None, scale=None):
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: _log_ndtr(d._z(v)), x)

    def logsf(self, x, loc=None, scale=None):
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: _log_ndtr(-d._z(v)), x)

    def isf(self, q, loc=None, scale=None):
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: d.loc - d.scale * _norm_ppf(v), q)

    def ppf(self, q, loc=None, scale=None):
        # either argument alone overrides the frozen value (scipy's
        # norm.cdf(x, scale=s) keeps loc = 0); the old test on loc only
        # silently dropped a scale passed without a loc
        d = self if loc is None and scale is None else _Norm(
            *self._ls(loc, scale))
        return _maybe_map(lambda v: d.loc + d.scale * _norm_ppf(v), q)

    @staticmethod
    def fit(data, *args, **kw):
        v = [float(x) for x in _flatten(data)]
        m = _math.fsum(v) / len(v)
        return (m, _math.sqrt(_math.fsum((x - m) ** 2 for x in v) / len(v)))

    def rvs(self, loc=None, scale=None, size=None, random_state=None):
        # Box-Muller through the generator; random_state may itself be a
        # generator (default_rng() used to hand one to SplitMix64's seed
        # arithmetic and die on `generator & mask`). loc/scale, as in
        # scipy norm.rvs(loc=, scale=, size=), override the frozen ones.
        return _rng_from(random_state).normal(
            self.loc if loc is None else loc,
            self.scale if scale is None else scale, size)


class _Chi2(_Dist):
    _support = (0.0, _math.inf)
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

    def sf(self, x, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(
            lambda v: 1.0 if v <= 0 else _gammainc_q(k / 2.0, v / 2.0), x)

    def logpdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if v < 0:
                return -_math.inf
            if v == 0:
                return _math.inf if k < 2 else (_math.log(0.5) if k == 2 else -_math.inf)
            return ((k / 2 - 1) * _math.log(v) - v / 2
                    - (k / 2) * _math.log(2) - _math.lgamma(k / 2))
        return _maybe_map(one, x)

    def ppf(self, q, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(
            lambda v: _ppf_from_cdf(
                lambda t: 0.0 if t <= 0 else _gammainc_p(k / 2, t / 2),
                v, 0.0, k + 10.0,
                sf=lambda t: 1.0 if t <= 0 else _gammainc_q(k / 2, t / 2)), q)

    def mean(self, df=None):
        return self.df if df is None else float(df)

    def var(self, df=None):
        return 2.0 * (self.df if df is None else float(df))

    def logsf(self, x, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(lambda v: 0.0 if v <= 0 else _log_gammainc_q(k / 2.0, v / 2.0), x)


def _t_cornish_fisher(x, nu):
    """Student t quantile from the normal quantile x, Abramowitz and
    Stegun 26.7.5 to fourth order; the error is O(nu^-5), below double
    precision for nu >= 1e5 and |x| <= 10 (checked against scipy)."""
    x2 = x * x
    g1 = (x2 + 1.0) * x / 4.0
    g2 = ((5.0 * x2 + 16.0) * x2 + 3.0) * x / 96.0
    g3 = (((3.0 * x2 + 19.0) * x2 + 17.0) * x2 - 15.0) * x / 384.0
    g4 = ((((79.0 * x2 + 776.0) * x2 + 1482.0) * x2 - 1920.0) * x2
          - 945.0) * x / 92160.0
    return x + (g1 + (g2 + (g3 + g4 / nu) / nu) / nu) / nu


class _T(_Dist):
    """Student t. df = inf is the standard normal, as in scipy."""

    def __init__(self, df=1.0):
        self.df = float(df)

    def pdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if k == _math.inf:
                return _math.exp(-0.5 * v * v) / _math.sqrt(2.0 * _math.pi)
            # Gamma((k+1)/2) / (sqrt(k pi) Gamma(k/2)) = 1 / (sqrt(k) B(1/2, k/2))
            ln = (-0.5 * _math.log(k) - _lbeta(0.5, k / 2.0)
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
            if k == _math.inf:
                return 0.5 * _math.erfc(-v / _math.sqrt(2.0))
            if k >= 1e5 and _bi.abs(v) <= 10.0:
                # huge df: invert the Cornish-Fisher quantile by Newton
                # with the exact density, from the normal start; the
                # incomplete-beta continued fraction loses ~df * eps here
                lower = v < 0
                w = v if lower else -v
                p_ = 0.5 * _math.erfc(-w / _math.sqrt(2.0))
                for _ in range(8):
                    step = (w - _t_cornish_fisher(float(norm.ppf(p_)), k)) \
                        * self.pdf(w, df=k)
                    p_ += step
                    if _bi.abs(step) <= 1e-17 * p_:
                        break
                return p_ if lower else 1.0 - p_
            # k / (k + v^2) without overflowing v^2 at |v| > 1e154
            r = _math.sqrt(k) / _bi.abs(v)
            if r < 1e-100:
                # far tail: I_x(k/2, 1/2) = r^k / ((k/2) B(k/2, 1/2)) (1 + O(r^2)),
                # in logs so it survives where r^k underflows
                ib = _math.exp(k * _math.log(r) - _math.log(k / 2.0) - _lbeta(k / 2.0, 0.5))
            else:
                ib = _betainc(k / 2.0, 0.5, r * r / (1.0 + r * r) if r < 1e150 else 1.0)
            return 1.0 - 0.5 * ib if v > 0 else 0.5 * ib
        return _maybe_map(one, x)

    def ppf(self, q, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if v == 0.5:
                return 0.0
            if v > 0.5:                 # symmetry: invert the upper half
                return -one(1.0 - v)
            if k == _math.inf:
                return float(norm.ppf(v))
            if k >= 1e5:
                x = float(norm.ppf(v))
                if _bi.abs(x) <= 10.0:
                    return _t_cornish_fisher(x, k)
            return _ppf_from_cdf(lambda u: self.cdf(u, df=k), v,
                                 -50.0 - _bi.min(k, 1e6), 0.0)
        return _maybe_map(one, q)

    def sf(self, x, df=None):
        """Upper tail by symmetry, sf(v) = cdf(-v): no 1 - cdf loss."""
        k = self.df if df is None else float(df)
        return _maybe_map(lambda v: self.cdf(-v, df=k), x)

    def logpdf(self, x, df=None):
        k = self.df if df is None else float(df)

        def one(v):
            if k == _math.inf:
                return -0.5 * v * v - 0.5 * _math.log(2.0 * _math.pi)
            return (-0.5 * _math.log(k) - _lbeta(0.5, k / 2.0)
                    - (k + 1) / 2 * _math.log1p(v * v / k))
        return _maybe_map(one, x)

    def isf(self, q, df=None):
        k = self.df if df is None else float(df)
        return _maybe_map(lambda v: -float(self.ppf(v, df=k)), q)

    def mean(self, df=None):
        k = self.df if df is None else float(df)
        return 0.0 if k > 1.0 else _math.nan

    def var(self, df=None):
        k = self.df if df is None else float(df)
        if k == _math.inf:
            return 1.0
        if k > 2.0:
            return k / (k - 2.0)
        return _math.inf if k > 1.0 else _math.nan


class _F(_Dist):
    _support = (0.0, _math.inf)
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
            # d2 log d2 - d2 log(d1 v + d2) = -d2 log1p(d1 v / d2): no
            # cancellation at large d2; the beta function via _lbeta
            ln = (0.5 * (d1 * _math.log(d1 * v) - d1 * _math.log(d1 * v + d2)
                         - d2 * _math.log1p(d1 * v / d2))
                  - _math.log(v)
                  - _lbeta(d1 / 2.0, d2 / 2.0))
            return _math.exp(ln)
        return _maybe_map(one, x)

    def sf(self, x, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)
        # I_{d2 / (d2 + d1 x)}(d2/2, d1/2): the small argument formed
        # directly, not as 1 - d1 x / (d1 x + d2), which rounds at large x
        return _maybe_map(
            lambda v: 1.0 if v <= 0 else _betainc(
                d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * v)), x)

    def logpdf(self, x, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)

        def one(v):
            if v <= 0:
                return -_math.inf
            return (0.5 * (d1 * _math.log(d1 * v) - d1 * _math.log(d1 * v + d2)
                           - d2 * _math.log1p(d1 * v / d2))
                    - _math.log(v) - _lbeta(d1 / 2.0, d2 / 2.0))
        return _maybe_map(one, x)

    def ppf(self, q, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)
        return _maybe_map(
            lambda v: _ppf_from_cdf(lambda t: float(self.cdf(t, d1, d2)), v,
                                    0.0, 10.0,
                                    sf=lambda t: float(self.sf(t, d1, d2))), q)

    def mean(self, dfn=None, dfd=None):
        d2 = self.dfd if dfd is None else float(dfd)
        return d2 / (d2 - 2.0) if d2 > 2.0 else _math.inf

    def var(self, dfn=None, dfd=None):
        d1 = self.dfn if dfn is None else float(dfn)
        d2 = self.dfd if dfd is None else float(dfd)
        if d2 > 4.0:
            return 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0))
        return _math.inf if d2 > 2.0 else _math.nan


def _digamma(x):
    """psi(x) for x > 0: recurrence up to 12, then the asymptotic series (truncation error below 1e-16 there)."""
    r = 0.0
    while x < 12.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + _math.log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (
        1.0 / 252 - f * (1.0 / 240 - f / 132))))


def _trigamma(x):
    """psi'(x) for x > 0: recurrence up to 12, then the asymptotic series (truncation error below 1e-16 there)."""
    r = 0.0
    while x < 12.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + 1.0 / x + f / 2.0 + (1.0 / x) * f * (1.0 / 6 - f * (
        1.0 / 30 - f * (1.0 / 42 - f / 30)))


class _Gamma(_Dist):
    _support = (0.0, _math.inf)
    def __init__(self, a=1.0, loc=0.0, scale=1.0):
        self.a, self.loc, self.scale = float(a), float(loc), float(scale)

    def fit(self, data, *args, floc=None, fscale=None, **kw):
        """Maximum likelihood with a fixed location (scipy's floc).

        With the location fixed, the shape solves
        log(a) - digamma(a) = log(mean) - mean(log x) and the scale is
        mean / a (Minka 2002); Newton steps from Minka's closed-form
        start converge in a handful of iterations. Returns
        (a, loc, scale) as scipy does. A free location is a different
        and often ill-posed problem, so it is refused rather than
        guessed.
        """
        del args, kw
        if floc is None:
            raise NotImplementedError(
                "gamma.fit needs floc= (a fixed location); the free-location "
                "fit is not implemented in the native core")
        x = [float(v) - float(floc) for v in _flatten(data)]
        if not x or any(not v > 0 for v in x):
            raise ValueError("gamma.fit: every observation must exceed floc")
        n = len(x)
        mean = _bi.sum(x) / n
        if fscale is not None:
            # shape given the scale solves digamma(a) = mean(log x) - log(scale)
            target = _bi.sum(_math.log(v) for v in x) / n - _math.log(float(fscale))
            a = 1.0
            for _ in range(200):
                step = (_digamma(a) - target) / _trigamma(a)
                a_new = a - step
                a = a_new if a_new > 0 else a / 2.0
                if abs(step) < 1e-14 * a:
                    break
            return (a, float(floc), float(fscale))
        s = _math.log(mean) - _bi.sum(_math.log(v) for v in x) / n
        if s <= 0:
            raise ValueError("gamma.fit: all observations are equal, so the "
                             "shape is unbounded")
        a = (3.0 - s + _math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        for _ in range(100):
            f = _math.log(a) - _digamma(a) - s
            fp = 1.0 / a - _trigamma(a)
            step = f / fp
            a_new = a - step
            a = a_new if a_new > 0 else a / 2.0
            if abs(step) < 1e-15 * a:
                break
        return (a, float(floc), mean / a)

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
            if z < 0:
                return 0.0
            if z == 0:
                # the density at the origin is z^(a-1)/Gamma(a)/scale:
                # unbounded for a < 1, 1/scale for a = 1 (exponential),
                # and 0 for a > 1, as scipy.stats.gamma gives
                if aa < 1:
                    return _math.inf
                if aa == 1:
                    return 1.0 / sc
                return 0.0
            ln = (aa - 1) * _math.log(z) - z - _math.lgamma(aa)
            return _math.exp(ln) / sc
        return _maybe_map(one, x)

    def sf(self, x, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)
        return _maybe_map(
            lambda v: 1.0 if v <= lo else _gammainc_q(aa, (v - lo) / sc), x)

    def logpdf(self, x, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)

        def one(v):
            z = (v - lo) / sc
            if z < 0:
                return -_math.inf
            if z == 0:
                return _math.inf if aa < 1 else (-_math.log(sc) if aa == 1 else -_math.inf)
            return (aa - 1) * _math.log(z) - z - _math.lgamma(aa) - _math.log(sc)
        return _maybe_map(one, x)

    def ppf(self, q, a=None, loc=0.0, scale=1.0):
        aa = self.a if a is None else float(a)
        lo = self.loc if a is None else float(loc)
        sc = self.scale if a is None else float(scale)
        return _maybe_map(
            lambda v: lo + sc * _ppf_from_cdf(
                lambda t: 0.0 if t <= 0 else _gammainc_p(aa, t),
                v, 0.0, aa + 10.0,
                sf=lambda t: 1.0 if t <= 0 else _gammainc_q(aa, t)), q)

    def mean(self, a=None, loc=0.0, scale=1.0):
        a = getattr(self, "a", 1.0) if a is None else float(a)
        return loc + a * scale

    def var(self, a=None, loc=0.0, scale=1.0):
        a = getattr(self, "a", 1.0) if a is None else float(a)
        return a * scale * scale

    def logsf(self, x, a=None, loc=0.0, scale=1.0):
        a = getattr(self, "a", 1.0) if a is None else float(a)
        return _maybe_map(lambda v: 0.0 if v <= loc else _log_gammainc_q(a, (v - loc) / scale), x)


class _Beta(_Dist):
    _support = (0.0, 1.0)

    def _ab(self, a, b):
        return (getattr(self, "a", None) if a is None else float(a),
                getattr(self, "b", None) if b is None else float(b))

    def mean(self, a=None, b=None):
        a, b = self._ab(a, b)
        return a / (a + b)

    def var(self, a=None, b=None):
        """Variance ab / ((a+b)^2 (a+b+1)) (Johnson, Kotz &
        Balakrishnan 1995, vol. 2, ch. 25)."""
        a, b = self._ab(a, b)
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    def std(self, a=None, b=None):
        return _math.sqrt(self.var(a, b))

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

    def sf(self, x, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)
        return _maybe_map(
            lambda v: _betaincc(aa, bb, min(max(v, 0.0), 1.0)), x)

    def logpdf(self, x, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)

        def one(v):
            if not 0.0 < v < 1.0:
                return -_math.inf
            return (-_lbeta(aa, bb) + (aa - 1) * _math.log(v)
                    + (bb - 1) * _math.log1p(-v))
        return _maybe_map(one, x)

    def ppf(self, q, a=None, b=None):
        aa = self.a if a is None else float(a)
        bb = self.b if b is None else float(b)
        return _maybe_map(
            lambda v: _ppf_from_cdf(lambda t: _betainc(aa, bb, min(max(t, 0.0), 1.0)), v,
                                    0.0, 1.0,
                                    sf=lambda t: _betaincc(aa, bb, min(max(t, 0.0), 1.0))), q)


class _Binom(_Dist):
    _discrete = True
    def _bounds(self, n=None, p=None):
        nn = n if n is not None else getattr(self, "_n", None)
        return (0.0, float(nn) if nn is not None else _math.inf)

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
        # P(X > k) = I_p(k + 1, n - k), formed directly (1 - cdf loses it)
        n, p = self._resolve(n, p)

        def one(kk):
            kk = int(_math.floor(kk))
            if kk < 0:
                return 1.0
            if kk >= n:
                return 0.0
            return _betainc(kk + 1.0, float(n - kk), p)
        return _maybe_map(one, k)


class _Poisson(_Dist):
    _support = (0.0, _math.inf)
    _discrete = True
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
        if _is_arraylike(mu):
            # scipy broadcasts k against mu: one rate per observation
            return _bcast(lambda kk, m: self.logpmf(kk, m), k, mu)
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

    @staticmethod
    def fit(data, *args, **kw):
        v = [float(x) for x in _flatten(data)]
        return (_bi.min(v), _bi.max(v) - _bi.min(v))

    def _bounds(self, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return (lo, lo + sc)

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

    def sf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: 1.0 if v <= lo else (0.0 if v >= lo + sc else (lo + sc - v) / sc), x)

    def logsf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)

        def one(v):
            if v <= lo:
                return 0.0
            if v >= lo + sc:
                return -_math.inf
            return _math.log1p(-(v - lo) / sc)
        return _maybe_map(one, x)

    def mean(self, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return lo + 0.5 * sc

    def var(self, loc=None, scale=None):
        sc = self.scale if scale is None else float(scale)
        return sc * sc / 12.0


class _Expon(_Dist):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    @staticmethod
    def fit(data, *args, **kw):
        v = [float(x) for x in _flatten(data)]
        lo = _bi.min(v)
        return (lo, _math.fsum(v) / len(v) - lo)

    def _bounds(self, loc=None, scale=None):
        return ((self.loc if loc is None else float(loc)), _math.inf)

    def pdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: _math.exp(-(v - lo) / sc) / sc if v >= lo else 0.0, x)

    def cdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(
            lambda v: 0.0 if v < lo else -_math.expm1(-(v - lo) / sc), x)

    def sf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: 1.0 if v < lo else _math.exp(-(v - lo) / sc), x)

    def logpdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: -(v - lo) / sc - _math.log(sc) if v >= lo else -_math.inf, x)

    def logsf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: -(v - lo) / sc if v >= lo else 0.0, x)

    def isf(self, q, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: lo - sc * _math.log(v), q)

    def ppf(self, q, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)
        return _maybe_map(lambda v: lo - sc * _math.log1p(-v), q)

    def logcdf(self, x, loc=None, scale=None):
        lo = self.loc if loc is None else float(loc)
        sc = self.scale if scale is None else float(scale)

        def one(v):
            if v < lo:
                return -_math.inf
            z = (v - lo) / sc
            # log(1 - e^-z): log(-expm1(-z)) near 0, log1p(-e^-z) far out
            return _math.log(-_math.expm1(-z)) if z < _math.log(2.0) else _math.log1p(-_math.exp(-z))
        return _maybe_map(one, x)

    def mean(self, loc=None, scale=None):
        return (self.loc if loc is None else float(loc)) + (self.scale if scale is None else float(scale))

    def var(self, loc=None, scale=None):
        sc = self.scale if scale is None else float(scale)
        return sc * sc


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
            raise ValueError(f"unsupported method {method!r}")
        i = j + 1
    from . import _array_core as _ac
    return _ac.marr(ranks)


def _pearson_r(x, y):
    mx, my = _mean(x), _mean(y)
    sxy = _math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = _math.fsum((a - mx) ** 2 for a in x)
    syy = _math.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    r = sxy / _math.sqrt(sxx * syy)
    return _bi.max(-1.0, _bi.min(1.0, r))


def _one_sided(p_two, stat, alternative):
    """scipy's alternative= on a symmetric two-sided p: half of it on the
    side the statistic falls, 1 - that half on the other side."""
    if alternative == "two-sided":
        return p_two
    half = p_two / 2.0
    if alternative == "greater":
        return half if stat >= 0 else 1.0 - half
    if alternative == "less":
        return half if stat <= 0 else 1.0 - half
    raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")


def pearsonr(x, y, alternative="two-sided"):
    x, y = _flatten(x), _flatten(y)
    n = len(x)
    r = _pearson_r(x, y)
    if r != r:
        return _TestResult(r, _math.nan)  # a constant input: undefined, as scipy
    if n < 3 or abs(r) == 1.0:
        return _TestResult(r, _one_sided(0.0 if abs(r) == 1.0 else 1.0, r, alternative))
    tstat = r * _math.sqrt((n - 2) / (1.0 - r * r))
    p = 2.0 * t.sf(abs(tstat), n - 2)
    return _TestResult(r, _one_sided(_bi.min(1.0, p), r, alternative))


def spearmanr(x, y, alternative="two-sided"):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry, alternative=alternative)


def pointbiserialr(x, y):
    return pearsonr(x, y)


def kendalltau(x, y, alternative="two-sided", variant="b", **kw):
    """Kendall's tau: ``variant="b"`` (tie-corrected, scipy default) or
    ``"c"`` (Stuart's tau-c for rectangular tables); the p-value uses
    the tau-b machinery in both cases, as scipy does."""
    del kw
    if variant not in ("b", "c"):
        raise ValueError("variant must be 'b' or 'c'")
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
    if variant == "c" and n > 1:
        m = min(len(set(x)), len(set(y)))
        tau = 2.0 * (conc - disc) / (n * n * (m - 1) / float(m)) if m > 1 else float("nan")
    if n < 2 or tau != tau:
        return _TestResult(tau, _math.nan)
    if n1 == 0 and n2 == 0 and n <= 50:
        # exact: the number of permutations of n items with k discordant
        # pairs (Mahonian numbers), two-sided as scipy
        counts = [1.0]
        for m in range(2, n + 1):
            new = [0.0] * (len(counts) + m - 1)
            run = 0.0
            for k in range(len(new)):
                run += counts[k] if k < len(counts) else 0.0
                if k - m >= 0:
                    run -= counts[k - m]
                new[k] = run
            counts = new
        total = _math.factorial(n)
        d = _bi.min(conc, disc)
        p_low = _math.fsum(counts[:d + 1]) / total
        return _TestResult(tau, _one_sided(_bi.min(1.0, 2.0 * p_low), tau, alternative))
    # asymptotic with the tie-corrected variance (Kendall 1970; scipy)
    def tie_sizes(v):
        c = {}
        for u in v:
            c[u] = c.get(u, 0) + 1
        return [k for k in c.values() if k > 1]
    tx_, ty_ = tie_sizes(x), tie_sizes(y)
    v0 = n * (n - 1) * (2 * n + 5)
    vt = _math.fsum(k * (k - 1) * (2 * k + 5) for k in tx_)
    vu = _math.fsum(k * (k - 1) * (2 * k + 5) for k in ty_)
    v1 = (_math.fsum(k * (k - 1) for k in tx_) * _math.fsum(k * (k - 1) for k in ty_)) \
        / (2.0 * n * (n - 1))
    v2 = (_math.fsum(k * (k - 1) * (k - 2) for k in tx_)
          * _math.fsum(k * (k - 1) * (k - 2) for k in ty_)) \
        / (9.0 * n * (n - 1) * (n - 2)) if n > 2 else 0.0
    var = (v0 - vt - vu) / 18.0 + v1 + v2
    z = (conc - disc) / _math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * norm.sf(abs(z))
    return _TestResult(tau, _one_sided(_bi.min(1.0, p), tau, alternative))


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
    intercept_stderr = stderr * _math.sqrt(_math.fsum(a * a for a in x) / n) \
        if n > 2 else 0.0
    return _LinregressResult(slope, intercept, r, p, stderr, intercept_stderr)


class _LinregressResult(tuple):
    """(slope, intercept, rvalue, pvalue, stderr) with attribute access,
    the shape scipy returns and every caller unpacks."""

    def __new__(cls, slope, intercept, rvalue, pvalue, stderr, intercept_stderr=0.0):
        obj = super().__new__(cls, (slope, intercept, rvalue, pvalue, stderr))
        obj.slope, obj.intercept, obj.rvalue = slope, intercept, rvalue
        obj.pvalue, obj.stderr, obj.intercept_stderr = pvalue, stderr, intercept_stderr
        obj.statistic = slope
        return obj


# ---------------------------------------------------- descriptive

def _by_axis(fn, a, axis):
    """Apply a 1-D statistic down `axis` of a 2-D input.

    skew and kurtosis took no `axis`, so every numpy-style call --
    fn/copod.py does stats.skew(X, axis=0) -- raised TypeError. Only
    axis=0 and axis=1 make sense for the 2-D inputs used here; None
    keeps the previous flatten-everything behaviour.
    """
    rows = [list(r) for r in a.tolist()] if hasattr(a, "tolist") else \
        [list(r) for r in a]
    if not rows or not isinstance(rows[0], list):
        return fn(rows)
    from . import _array_core as _ac2
    if axis == 0:
        cols = list(zip(*rows))
        return _ac2.marr([fn(list(c)) for c in cols])
    if axis == 1:
        return _ac2.marr([fn(list(r)) for r in rows])
    raise ValueError(f"axis must be 0, 1 or None, got {axis!r}")


def skew(a, bias=True, axis=None):
    if axis is not None:
        return _by_axis(lambda v: skew(v, bias=bias), a, axis)
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


def kurtosis(a, fisher=True, bias=True, axis=None):
    if axis is not None:
        return _by_axis(lambda v: kurtosis(v, fisher=fisher, bias=bias),
                        a, axis)
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


def iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy="propagate",
        interpolation="linear", keepdims=False):
    """scipy.stats.iqr over the whole sample (axis=None).

    ``interpolation`` follows numpy's percentile methods: linear, lower,
    higher, midpoint, nearest. ``scale`` divides the result; "normal"
    makes it a consistent estimator of a normal standard deviation.
    It used to accept only the sample, so any caller passing scipy's
    keywords raised TypeError.
    """
    del keepdims
    if axis is not None:
        raise NotImplementedError("iqr: axis is not supported; the native "
                                  "core computes over the whole sample")
    v = [float(t) for t in _flatten(x)]
    if any(_math.isnan(t) for t in v):
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        if nan_policy == "propagate":
            return _math.nan
        v = [t for t in v if not _math.isnan(t)]
    if not v:
        return _math.nan
    v.sort()
    n = len(v)
    lo_p, hi_p = float(rng[0]), float(rng[1])
    if not 0 <= lo_p <= hi_p <= 100:
        raise ValueError("rng must satisfy 0 <= rng[0] <= rng[1] <= 100")

    def q(pct):
        h = (n - 1) * pct / 100.0
        lo = int(_math.floor(h))
        hi = _bi.min(lo + 1, n - 1)
        frac = h - lo
        if interpolation == "linear":
            return v[lo] + frac * (v[hi] - v[lo])
        if interpolation == "lower":
            return v[lo]
        if interpolation == "higher":
            return v[hi] if frac > 0 else v[lo]
        if interpolation == "midpoint":
            return 0.5 * (v[lo] + v[hi]) if frac > 0 else v[lo]
        if interpolation == "nearest":
            # numpy rounds half to even on the fractional index
            return v[int(round(h))]
        raise ValueError("interpolation must be linear, lower, higher, "
                         "midpoint or nearest")
    out = q(hi_p) - q(lo_p)
    if isinstance(scale, str):
        if scale != "normal":
            raise ValueError("scale must be a number or 'normal'")
        # scipy uses the normal IQR, Phi^-1(0.75) - Phi^-1(0.25), for
        # every rng: "normal" rescales to a normal sd only at (25, 75)
        sc = norm.ppf(0.75) - norm.ppf(0.25)
    else:
        sc = float(scale)
    return out / sc


class _DescribeResult(tuple):
    """scipy's DescribeResult: the 6-tuple (nobs, minmax, mean, variance,
    skewness, kurtosis) with the same names as attributes."""

    _fields = ("nobs", "minmax", "mean", "variance", "skewness", "kurtosis")

    def __new__(cls, nobs, minmax, mean, variance, skewness, kurtosis):
        obj = super().__new__(cls, (nobs, minmax, mean, variance, skewness, kurtosis))
        for k, v in zip(cls._fields, obj):
            setattr(obj, k, v)
        return obj

    def __repr__(self):
        return "DescribeResult({})".format(", ".join(
            f"{k}={getattr(self, k)!r}" for k in self._fields))


def describe(a, ddof=1):
    v = _flatten(a)
    return _DescribeResult(len(v), (min(v), max(v)), _mean(v),
                           _var(v, ddof=ddof), skew(v), kurtosis(v))


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
    d = _mean(v) - float(popmean)
    # no spread: scipy returns t = +-inf (p = 0 on the matching side) for a
    # non-zero difference and nan when it is zero too
    no_spread = _math.copysign(float("inf"), d) if d != 0.0 else float("nan")
    stat = no_spread if se == 0.0 else d / se
    if stat != stat:
        return _TestResult(stat, float("nan"), df=n - 1)
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


def ttest_rel(a, b, alternative="two-sided", **kw):
    del kw
    x, y = _flatten(a), _flatten(b)
    if len(x) != len(y):
        raise ValueError("unequal length arrays")
    return ttest_1samp([u - w for u, w in zip(x, y)], 0.0,
                       alternative=alternative)


def _mwu_exact_counts(n1, n2):
    """Number of orderings giving each U in 0..n1*n2 (no ties):
    f(n1, n2, u) = f(n1 - 1, n2, u - n2) + f(n1, n2 - 1, u)."""
    prev = [[1] for _ in range(n2 + 1)]          # n1 = 0: U = 0 only
    for i in range(1, n1 + 1):
        cur = [[1]]                               # n2 = 0: U = 0 only
        for j in range(1, n2 + 1):
            size = i * j + 1
            row = [0] * size
            for u, c in enumerate(prev[j]):       # f(i-1, j, u - j)
                row[u + j] += c
            for u, c in enumerate(cur[j - 1]):    # f(i, j-1, u)
                row[u] += c
            cur.append(row)
        prev = cur
    return prev[n2]


def mannwhitneyu(x, y, alternative="two-sided", use_continuity=True,
                 method="auto", **kw):
    """scipy.stats.mannwhitneyu: the statistic is U1; 'auto' uses the
    exact null distribution when both samples have at most 8 values and
    there are no ties, else the tie-corrected normal approximation with
    an optional continuity correction on the side-appropriate U."""
    del kw
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    ranks = rankdata(xv + yv)
    r1 = _math.fsum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    counts = {}
    for v in xv + yv:
        counts[v] = counts.get(v, 0) + 1
    ties = _bi.any(c > 1 for c in counts.values())
    if method == "auto":
        method = "exact" if (n1 <= 8 and n2 <= 8 and not ties) else "asymptotic"
    if alternative == "greater":
        U = u1
    elif alternative == "less":
        U = u2
    else:
        U = _bi.max(u1, u2)
    if method == "exact":
        cnt = _mwu_exact_counts(n1, n2)
        tot = _math.fsum(cnt)
        k = int(round(U))
        p = _math.fsum(cnt[k:]) / tot             # P(U_null >= U)
    else:
        mu = n1 * n2 / 2.0
        n = n1 + n2
        tie = _math.fsum(c ** 3 - c for c in counts.values())
        sig = _math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
        z = (U - mu - (0.5 if use_continuity else 0.0)) / sig
        p = norm.sf(z)
    if alternative == "two-sided":
        p = 2.0 * p
    return _TestResult(u1, _bi.min(1.0, _bi.max(0.0, p)))


def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", method="auto", **kw):
    """Wilcoxon signed-rank test with scipy.stats.wilcoxon's rules.

    ``method="auto"`` is exact when there are no ties and no zeros and
    n <= 50; with ties or zeros it enumerates all 2^n sign flips (the
    permutation test scipy runs) when n <= 13, and uses the tie-corrected
    normal approximation otherwise.  The statistic is min(W+, W-) for a
    two-sided test and W+ for a one-sided one.
    """
    del kw
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    if zero_method not in ("wilcox", "pratt"):
        raise ValueError("zero_method must be 'wilcox' or 'pratt'")
    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be 'auto', 'exact' or 'asymptotic'")
    xv = _flatten(x)
    if y is not None:
        yv = _flatten(y)
        if len(yv) != len(xv):
            raise ValueError("x and y must have the same length")
        d = [a - b for a, b in zip(xv, yv)]
    else:
        d = list(xv)
    d = [v for v in d if v == v]
    n_all = len(d)
    if n_all == 0:
        return _TestResult(_math.nan, _math.nan)
    n_zero = sum(1 for v in d if v == 0.0)
    keep = d if zero_method == "pratt" else [v for v in d if v != 0.0]
    count = len(keep)
    ranks = rankdata([abs(v) for v in keep]) if keep else []
    wplus = _math.fsum(r for r, v in zip(ranks, keep) if v > 0)
    wminus = _math.fsum(r for r, v in zip(ranks, keep) if v < 0)
    stat = _bi.min(wplus, wminus) if alternative == "two-sided" else wplus
    groups = {}
    for v in keep:
        if v != 0.0:
            groups[abs(v)] = groups.get(abs(v), 0) + 1
    has_ties = any(c > 1 for c in groups.values())
    if method == "auto":
        if n_all > 50:
            method = "asymptotic"
        elif not (has_ties or n_zero > 0):
            method = "exact"
        elif n_all <= 13:
            method = "permutation"
        else:
            method = "asymptotic"
    if method in ("exact", "permutation"):
        # null distribution of W+ over the 2^m sign flips of the non-zero
        # differences, on doubled ranks so mid-ranks stay integral
        r2 = [int(round(2.0 * r)) for r, v in zip(ranks, keep) if v != 0.0]
        total = sum(r2)
        dist = [0.0] * (total + 1)
        dist[0] = 1.0
        for r in r2:
            for w in range(total, r - 1, -1):
                dist[w] += dist[w - r]
        scale = 2.0 ** len(r2)
        w2 = 2.0 * wplus
        if method == "exact":
            # scipy rounds a non-integral W+ up for the cdf and down for the sf
            lo_k = int(_math.ceil(w2 / 2.0 - 1e-12)) * 2
            hi_k = int(_math.floor(w2 / 2.0 + 1e-12)) * 2
        else:
            lo_k = int(_math.floor(w2 + 1e-9))
            hi_k = int(_math.ceil(w2 - 1e-9))
        p_le = _math.fsum(dist[:_bi.max(0, _bi.min(lo_k, total)) + 1]) / scale if lo_k >= 0 else 0.0
        p_ge = _math.fsum(dist[_bi.max(0, hi_k):]) / scale if hi_k <= total else 0.0
        if alternative == "less":
            p = p_le
        elif alternative == "greater":
            p = p_ge
        else:
            p = _bi.min(1.0, 2.0 * _bi.min(p_le, p_ge))
        return _TestResult(stat, p)
    mn = count * (count + 1) / 4.0
    var = count * (count + 1) * (2 * count + 1)
    if zero_method == "pratt":
        mn -= n_zero * (n_zero + 1) / 4.0
        var -= n_zero * (n_zero + 1) * (2 * n_zero + 1)
    tie = _math.fsum(c ** 3 - c for c in groups.values())
    se = _math.sqrt((var - tie / 2.0) / 24.0) if var - tie / 2.0 > 0 else 0.0
    if se == 0.0:
        return _TestResult(stat, _math.nan)
    z = (wplus - mn) / se
    if correction:
        sgn = 1.0 if alternative == "greater" else -1.0 if alternative == "less" \
            else (1.0 if z > 0 else -1.0 if z < 0 else 0.0)
        z -= sgn * 0.5 / se
    if alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        p = _bi.min(1.0, 2.0 * norm.sf(abs(z)))
    return _TestResult(stat, p)


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


_CRESSIE_READ = {"pearson": 1.0, "log-likelihood": 0.0,
                 "freeman-tukey": -0.5, "mod-log-likelihood": -1.0,
                 "neyman": -2.0, "cressie-read": 2.0 / 3.0}


def _power_divergence_stat(obs, exp, lambda_):
    """Cressie-Read statistic for one pair of flat sequences."""
    lam = _CRESSIE_READ[lambda_] if isinstance(lambda_, str) else (
        1.0 if lambda_ is None else float(lambda_))
    if lam == 1.0:
        return _math.fsum((o - e) ** 2 / e for o, e in zip(obs, exp))
    if lam == 0.0:
        return 2.0 * _math.fsum(o * _math.log(o / e)
                                for o, e in zip(obs, exp) if o > 0)
    if lam == -1.0:
        return 2.0 * _math.fsum(e * _math.log(e / o)
                                for o, e in zip(obs, exp) if o > 0)
    return (2.0 / (lam * (lam + 1.0))) * _math.fsum(
        o * ((o / e) ** lam - 1.0) for o, e in zip(obs, exp))


def chi2_contingency(observed, correction=True, lambda_=None):
    """Chi-square (or any Cressie-Read power-divergence) test of
    independence. With ``correction`` and one degree of freedom the
    observed counts are shifted half a unit toward the expected ones,
    as scipy does, before the statistic is formed."""
    if hasattr(observed, "columns") and hasattr(observed, "values"):
        observed = observed.values          # scipy: np.asarray(frame)
    rows = observed.tolist() if hasattr(observed, "tolist") \
        else [list(r) for r in observed]
    rows = [[float(v) for v in r] for r in rows]
    r, c = len(rows), len(rows[0])
    rt = [_math.fsum(row) for row in rows]
    ct = [_math.fsum(rows[i][j] for i in range(r)) for j in range(c)]
    n = _math.fsum(rt)
    exp = [[rt[i] * ct[j] / n for j in range(c)] for i in range(r)]
    dof = (r - 1) * (c - 1)
    obs = [rows[i][j] for i in range(r) for j in range(c)]
    e = [exp[i][j] for i in range(r) for j in range(c)]
    if correction and dof == 1:
        adj = []
        for o, ev in zip(obs, e):
            diff = ev - o
            mag = _bi.min(0.5, abs(diff))
            adj.append(o + (mag if diff > 0 else -mag if diff < 0 else 0.0))
        obs = adj
    stat = _power_divergence_stat(obs, e, lambda_)
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
    # scipy's statistic is the observed proportion k / n, not the count
    return _TestResult(k / n, _bi.min(1.0, pv),
                       k=k, n=n, proportion_estimate=k / n,
                       proportion_ci=lambda confidence_level=0.95, method="exact":
                       _proportion_ci(k, n, confidence_level, method))


class _ConfidenceInterval(tuple):
    def __new__(cls, low, high):
        obj = super().__new__(cls, (low, high))
        obj.low, obj.high = low, high
        return obj


def _proportion_ci(k, n, confidence_level=0.95, method="exact"):
    """Confidence interval for a binomial proportion.

    ``exact`` is Clopper-Pearson through the beta quantiles
    (low = B(a/2; k, n-k+1), high = B(1-a/2; k+1, n-k), with 0 and 1 at
    the ends); ``wilson`` is the score interval. This is what
    scipy's BinomTestResult.proportion_ci returns, and what
    mrm_oneprop_test() called on a result that did not have it.
    """
    a = 1.0 - float(confidence_level)
    k, n = int(k), int(n)
    if method == "exact":
        low = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
        high = 1.0 if k == n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
        return _ConfidenceInterval(low, high)
    if method == "wilson":
        z = _norm_ppf(1.0 - a / 2.0)
        ph = k / n
        den = 1.0 + z * z / n
        centre = (ph + z * z / (2.0 * n)) / den
        half = z * _math.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n)) / den
        return _ConfidenceInterval(_bi.max(0.0, centre - half), _bi.min(1.0, centre + half))
    raise ValueError("method must be 'exact' or 'wilson'")


# ---------------------------------------------------- KS family

def _ks_sf(d, n):
    """Two-sided asymptotic Kolmogorov Q(d*sqrt(n)) w/ Stephens correction.

    Stephens, M. A. (1970) "Use of the Kolmogorov-Smirnov, Cramer-von
    Mises and related statistics without extensive tables", *Journal of
    the Royal Statistical Society, Series B* 32(1), 115-122,
    doi:10.1111/j.2517-6161.1970.tb00821.x -- the small-sample
    correction applied to the asymptotic series.
    """
    lam = d * (_math.sqrt(n) + 0.12 + 0.11 / _math.sqrt(n))
    if lam < 0.04:
        # Q(lam) -> 1 as lam -> 0, but the alternating series below does
        # not converge there: its terms stay near 2 and the truncated
        # sum lands on an arbitrary value (0.0 at lam = 0, 0.35 at
        # lam = 0.005).  Return the limit instead.  This is the branch
        # taken when two samples are identical, i.e. D = 0 with ties, in
        # which case the correct p-value is 1.
        return 1.0
    s = 0.0
    for j in range(1, 101):
        term = 2.0 * (-1) ** (j - 1) * _math.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) < 1e-12:
            break
    return _bi.max(0.0, _bi.min(1.0, s))


def _ks_pkolmogorov(d, n):
    """P(D_n < d), exact, by Marsaglia, Tsang and Wang (2003).

    Marsaglia, G., Tsang, W. W. & Wang, J. (2003) "Evaluating
    Kolmogorov's distribution", *Journal of Statistical Software*
    8(18), 1-4, doi:10.18637/jss.v008.i18.

    Journal of Statistical Software 8(18), "Evaluating Kolmogorov's
    Distribution".  The same algorithm R's ks.test uses for n < 100.
    H is (2k-1) square with k = ceil(n d); the answer is n! / n^n times
    the (k,k) entry of H^n, and the factorial is folded into the matrix
    power in blocks so the intermediate entries cannot overflow.
    """
    d = float(d)
    n = int(n)
    if d <= 0.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    H = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i - j + 1 >= 0:
                H[i][j] = 1.0
    for i in range(m):
        H[i][0] -= h ** (i + 1)
        H[m - 1][i] -= h ** (m - i)
    H[m - 1][0] += (2.0 * h - 1.0) ** m if 2.0 * h - 1.0 > 0.0 else 0.0
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 2):
                    H[i][j] /= g

    def mul(A, B):
        return [[_math.fsum(A[i][t] * B[t][j] for t in range(m))
                 for j in range(m)] for i in range(m)]

    # binary exponentiation, rescaling by 2^-128 whenever entries grow
    eQ = 0
    Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    P = [row[:] for row in H]
    eP = 0
    e = n
    while e > 0:
        if e & 1:
            Q = mul(Q, P)
            eQ += eP
            if Q[k - 1][k - 1] > 1e140:
                Q = [[v * 1e-140 for v in row] for row in Q]
                eQ += 140
        e >>= 1
        if e:
            P = mul(P, P)
            eP *= 2
            if P[k - 1][k - 1] > 1e140:
                P = [[v * 1e-140 for v in row] for row in P]
                eP += 140
    val = Q[k - 1][k - 1]
    for i in range(1, n + 1):
        val *= i / n
        if val < 1e-140:
            val *= 1e140
            eQ -= 140
    return val * 10.0 ** eQ


def _ks_psmirnov(d, n1, n2, two_sided=True):
    """P(D < d) for the two-sample statistic, exact, no ties.

    The recursion R's ks.test uses (psmirnov2x): count the lattice paths
    from (0,0) to (n1,n2) that never leave the band |i/n1 - j/n2| <= q,
    carrying the hypergeometric weights along.  With ``two_sided`` off
    only the upper edge of the band constrains the path, which is the
    one-sided law of D+.
    """
    md, nd = float(n1), float(n2)
    q = (0.5 + _math.floor(float(d) * md * nd - 1e-7)) / (md * nd)
    u = [0.0] * (n2 + 1)
    for j in range(n2 + 1):
        u[j] = 0.0 if (two_sided and (j / nd) > q) else 1.0
    for i in range(1, n1 + 1):
        w = i / (i + nd)
        u[0] = 0.0 if (i / md) > q else w * u[0]
        for j in range(1, n2 + 1):
            gap = i / md - j / nd
            if (abs(gap) if two_sided else gap) > q:
                u[j] = 0.0
            else:
                u[j] = w * u[j] + u[j - 1]
    return u[n2]


_KS_ALTERNATIVES = ("two-sided", "less", "greater")


def _ks_check_alt(alternative):
    if alternative not in _KS_ALTERNATIVES:
        raise ValueError("alternative must be one of {}, got {!r}".format(", ".join(_KS_ALTERNATIVES), alternative))
    return alternative


def ks_1samp(x, cdf, args=(), alternative="two-sided"):
    """One-sample Kolmogorov-Smirnov statistic and p-value.

    ``alternative`` names the alternative hypothesis in terms of the
    CDFs, the same convention R's ks.test uses:

      "two-sided"  F != G, statistic D  = max(D+, D-)
      "greater"    F >  G, statistic D+ = max(ECDF - CDF)
      "less"       F <  G, statistic D- = max(CDF - ECDF)

    The one-sided p-values are EXACT (Birnbaum-Tingey).  The two-sided
    one is the asymptotic Kolmogorov series with Stephens' small-sample
    correction, which is why ``exact`` is reported: at small n a
    two-sided p-value near the decision boundary should not be leaned on.
    """
    _ks_check_alt(alternative)
    v = sorted(_flatten(x))
    n = len(v)
    if n < 1:
        raise ValueError("need at least one observation")
    cdfv = [float(cdf(u, *args)) if callable(cdf) else float(u) for u in v]
    dplus = _bi.max((i + 1) / n - cdfv[i] for i in range(n))
    dminus = _bi.max(cdfv[i] - i / n for i in range(n))
    dplus = _bi.max(0.0, dplus)
    dminus = _bi.max(0.0, dminus)
    if alternative == "greater":
        d, pv, exact = dplus, _KSOne.sf(dplus, n), True
    elif alternative == "less":
        d, pv, exact = dminus, _KSOne.sf(dminus, n), True
    else:
        d = _bi.max(dplus, dminus)
        if n < 100:
            pv = _bi.max(0.0, _bi.min(1.0, 1.0 - _ks_pkolmogorov(d, n)))
            exact = True
        else:
            pv, exact = _ks_sf(d, n), False
    return _TestResult(d, pv, n=n, d_plus=dplus, d_minus=dminus,
                       alternative=alternative, exact=exact)


def kstest(rvs, cdf, args=(), alternative="two-sided"):
    _ks_check_alt(alternative)
    if isinstance(cdf, str):
        try:
            dist = {"norm": norm, "uniform": uniform, "expon": expon}[cdf]
        except KeyError:
            raise ValueError(
                f"unknown distribution {cdf!r}; known: norm, uniform, expon") from None
        return ks_1samp(rvs, lambda u, *a: dist.cdf(u, *a), args,
                        alternative=alternative)
    if callable(cdf):
        return ks_1samp(rvs, cdf, args, alternative=alternative)
    return ks_2samp(rvs, cdf, alternative=alternative)


def ks_2samp(a, b, alternative="two-sided"):
    """Two-sample Kolmogorov-Smirnov (Smirnov) statistic and p-value.

    The one-sided p-value is Smirnov's asymptotic exp(-2 n_e D^2); the
    two-sided one is the Kolmogorov series in the effective sample size
    n_e = n1 n2 / (n1 + n2), used above n1 n2 = 10000; below that the
    exact Smirnov distribution is used whether or not there are ties, as
    scipy does (it is conservative with ties, whose count is reported).
    """
    _ks_check_alt(alternative)
    x, y = sorted(_flatten(a)), sorted(_flatten(b))
    n1, n2 = len(x), len(y)
    if n1 < 1 or n2 < 1:
        raise ValueError("both samples need at least one observation")
    allv = sorted(set(x + y))
    ties = (n1 + n2) - len(allv)
    is_exact = False

    def ecdf(sorted_v, u):
        import bisect
        return bisect.bisect_right(sorted_v, u) / len(sorted_v)

    diffs = [ecdf(x, u) - ecdf(y, u) for u in allv]
    dplus = _bi.max(0.0, _bi.max(diffs))
    dminus = _bi.max(0.0, -_bi.min(diffs))
    en = n1 * n2 / (n1 + n2)
    if alternative in ("greater", "less"):
        d = dplus if alternative == "greater" else dminus
        if n1 * n2 <= 10000:
            pv = 1.0 - _ks_psmirnov(d, n1, n2, two_sided=False)
            is_exact = True
        else:
            pv = _math.exp(-2.0 * en * d * d)
    else:
        d = _bi.max(dplus, dminus)
        if n1 * n2 <= 10000:
            pv = 1.0 - _ks_psmirnov(d, n1, n2)
            is_exact = True
        else:
            pv = _ks_sf(d, en)
    return _TestResult(d, _bi.max(0.0, _bi.min(1.0, pv)),
                       n1=n1, n2=n2, d_plus=dplus, d_minus=dminus,
                       alternative=alternative, exact=is_exact, n_ties=ties)


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


def _gumbel_r_fit(v):
    """MLE of the right-skewed Gumbel: the scale solves
    mean(x) - sum(x e^{-x/b}) / sum(e^{-x/b}) = b (bisection), then
    loc = -b log(mean(e^{-x/b}))."""
    n = len(v)
    mu = _math.fsum(v) / n

    def h(b):
        e = [-(u - mu) / b for u in v]
        m = max(e)
        w = [_math.exp(t - m) for t in e]
        sw = _math.fsum(w)
        return mu - _math.fsum(u * wi for u, wi in zip(v, w)) / sw - b
    sd = _math.sqrt(_var(v, ddof=1)) or 1e-8
    lo, hi = sd * 1e-3, sd * 10.0
    flo, fhi = h(lo), h(hi)
    for _ in range(60):
        if flo * fhi > 0:
            hi *= 2.0
            fhi = h(hi)
            continue
        mid = 0.5 * (lo + hi)
        fm = h(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    b = 0.5 * (lo + hi)
    e = [-(u - mu) / b for u in v]
    m = max(e)
    loc = mu - b * (m + _math.log(_math.fsum(_math.exp(t - m) for t in e) / n))
    return loc, b


def _logistic_fit(v):
    """MLE of the logistic distribution by Newton steps on the score
    equations sum tanh(z/2) = 0 and sum z tanh(z/2) = n, z = (x-loc)/s."""
    n = len(v)
    loc = _math.fsum(v) / n
    s = _math.sqrt(_var(v, ddof=1)) * _math.sqrt(3.0) / _math.pi or 1e-8

    def score(loc, s):
        z = [(u - loc) / s for u in v]
        t = [_math.tanh(zi / 2.0) for zi in z]
        return (_math.fsum(t), _math.fsum(zi * ti for zi, ti in zip(z, t)) - n)
    for _ in range(100):
        f1, f2 = score(loc, s)
        if abs(f1) < 1e-10 * n and abs(f2) < 1e-10 * n:
            break
        h1, h2 = 1e-6 * (abs(loc) + 1.0), 1e-6 * s
        a11 = (score(loc + h1, s)[0] - f1) / h1
        a12 = (score(loc, s + h2)[0] - f1) / h2
        a21 = (score(loc + h1, s)[1] - f2) / h1
        a22 = (score(loc, s + h2)[1] - f2) / h2
        det = a11 * a22 - a12 * a21
        if det == 0:
            break
        dl = (f1 * a22 - f2 * a12) / det
        ds = (a11 * f2 - a21 * f1) / det
        loc -= dl
        s = max(s - ds, 1e-12)
    return loc, s


def anderson(x, dist="norm"):
    """Anderson-Darling test for a fitted normal, exponential, logistic
    or Gumbel distribution, with scipy's critical-value tables and
    finite-sample corrections (Stephens 1974 / D'Agostino-Stephens 1986)."""
    v = sorted(_flatten(x))
    n = len(v)
    if n < 2:
        raise ValueError("anderson needs at least two observations")
    if dist == "norm":
        mu, sd = _mean(v), _math.sqrt(_var(v, ddof=1))
        z = [norm.cdf((u - mu) / sd) for u in v]
        base = [0.561, 0.631, 0.752, 0.873, 1.035]  # scipy 1.18 table
        crit = [round(b / (1.0 + 0.75 / n + 2.25 / (n * n)), 3) for b in base]
        sig = [15.0, 10.0, 5.0, 2.5, 1.0]
        fit = (mu, sd)
    elif dist == "expon":
        scale = _mean(v)
        z = [-_math.expm1(-u / scale) for u in v]
        base = [0.916, 1.062, 1.321, 1.591, 1.959]  # scipy 1.18 table
        crit = [round(b / (1.0 + 0.6 / n), 3) for b in base]
        sig = [15.0, 10.0, 5.0, 2.5, 1.0]
        fit = (0.0, scale)
    elif dist == "logistic":
        loc, s = _logistic_fit(v)
        z = [1.0 / (1.0 + _math.exp(-(u - loc) / s)) for u in v]
        base = [0.426, 0.563, 0.660, 0.769, 0.906, 1.010]
        crit = [round(b / (1.0 + 0.25 / n), 3) for b in base]
        sig = [25.0, 10.0, 5.0, 2.5, 1.0, 0.5]
        fit = (loc, s)
    elif dist in ("gumbel", "gumbel_l", "extreme1", "gumbel_r"):
        if dist == "gumbel_r":
            loc, b = _gumbel_r_fit(v)
            z = [_math.exp(-_math.exp(-(u - loc) / b)) for u in v]
        else:
            loc_n, b = _gumbel_r_fit([-u for u in v])
            loc = -loc_n
            z = [-_math.expm1(-_math.exp((u - loc) / b)) for u in v]
        base = [0.474, 0.637, 0.757, 0.877, 1.038]
        crit = [round(b_ / (1.0 + 0.2 / _math.sqrt(n)), 3) for b_ in base]
        sig = [25.0, 10.0, 5.0, 2.5, 1.0]
        fit = (loc, b)
    else:
        raise ValueError("Invalid distribution; dist must be 'norm', "
                         "'expon', 'gumbel', 'gumbel_l', 'gumbel_r', "
                         "'extreme1' or 'logistic'.")
    eps = 1e-300
    a2 = -n - _math.fsum(
        (2 * (i + 1) - 1) * (_math.log(max(z[i], eps))
                             + _math.log(max(1.0 - z[n - 1 - i], eps)))
        for i in range(n)) / n
    from . import _array_core as _ac
    return _TestResult(a2, None,
                       critical_values=_ac.marr(crit),
                       significance_level=_ac.marr(sig),
                       fit_result=fit)


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
        if isinstance(points, (int, float)):
            points = [float(points)]
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
            z = (v - loc) / scale
            return 1.0 / (1.0 + _math.exp(-z)) if z >= 0 else _math.exp(z) / (1.0 + _math.exp(z))
        return _maybe_map(one, x)

    def sf(self, x, loc=0.0, scale=1.0):
        return self.cdf(_maybe_map(lambda v: 2.0 * loc - v, x), loc, scale)

    def logcdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return -_math.log1p(_math.exp(-z)) if z >= 0 else z - _math.log1p(_math.exp(z))
        return _maybe_map(one, x)

    def logsf(self, x, loc=0.0, scale=1.0):
        return self.logcdf(_maybe_map(lambda v: 2.0 * loc - v, x), loc, scale)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return -_math.inf
            if p == 1.0:
                return _math.inf
            return loc + scale * (_math.log(p) - _math.log1p(-p))
        return _maybe_map(one, q)

    def isf(self, q, loc=0.0, scale=1.0):
        return _maybe_map(lambda p: 2.0 * loc - _scalar(self.ppf(p, loc, scale)), q)

    def mean(self, loc=0.0, scale=1.0):
        return float(loc)

    def var(self, loc=0.0, scale=1.0):
        return (_math.pi * scale) ** 2 / 3.0


class _Laplace(_Dist):
    @staticmethod
    def fit(data, *args, **kw):
        v = sorted(float(x) for x in _flatten(data))
        n = len(v)
        med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
        return (med, _math.fsum(abs(x - med) for x in v) / n)

    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            return _math.exp(-abs(v - loc) / scale) / (2.0 * scale)
        return _maybe_map(one, x)


    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            return loc + scale * (_math.log(2.0 * p) if p < 0.5
                                  else -_math.log(2.0 * (1.0 - p)))
        return _maybe_map(one, q)

    def sf(self, x, loc=0.0, scale=1.0):
        return self.cdf(_maybe_map(lambda v: 2.0 * loc - v, x), loc, scale)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.5 * _math.exp(z) if z < 0 else 1.0 - 0.5 * _math.exp(-z)
        return _maybe_map(one, x)

    def logcdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return _math.log(0.5) + z if z < 0 else _math.log1p(-0.5 * _math.exp(-z))
        return _maybe_map(one, x)

    def logsf(self, x, loc=0.0, scale=1.0):
        return self.logcdf(_maybe_map(lambda v: 2.0 * loc - v, x), loc, scale)

    def isf(self, q, loc=0.0, scale=1.0):
        return _maybe_map(lambda p: 2.0 * loc - _scalar(self.ppf(p, loc, scale)), q)

    def mean(self, loc=0.0, scale=1.0):
        return float(loc)

    def var(self, loc=0.0, scale=1.0):
        return 2.0 * scale * scale


class _Cauchy(_Dist):
    # no finite moments: scipy reports nan, and the entropy in closed form
    def mean(self, loc=0.0, scale=1.0):
        return _math.nan

    def var(self, loc=0.0, scale=1.0):
        return _math.nan

    def std(self, loc=0.0, scale=1.0):
        return _math.nan

    def moment(self, order, loc=0.0, scale=1.0):
        return _math.nan

    def entropy(self, loc=0.0, scale=1.0):
        return _math.log(4.0 * _math.pi * scale)

    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 1.0 / (_math.pi * scale * (1.0 + z * z))
        return _maybe_map(one, x)


    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z != z:
                return _math.nan
            # 1/2 + atan(z)/pi cancels for z << 0: atan(-1/z)/pi there
            return _math.atan(-1.0 / z) / _math.pi if z < -1.0 else 0.5 + _math.atan(z) / _math.pi
        return _maybe_map(one, x)

    def sf(self, x, loc=0.0, scale=1.0):
        return self.cdf(_maybe_map(lambda v: 2.0 * loc - v, x), loc, scale)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return -_math.inf
            if p == 1.0:
                return _math.inf
            if p == 0.5:
                return float(loc)
            # tan(pi (p - 1/2)) = -1/tan(pi p): exact in p for small p
            return loc - scale / _math.tan(_math.pi * p) if p < 0.5 \
                else loc + scale / _math.tan(_math.pi * (1.0 - p))
        return _maybe_map(one, q)

    def isf(self, q, loc=0.0, scale=1.0):
        return _maybe_map(lambda p: 2.0 * loc - _scalar(self.ppf(p, loc, scale)), q)


class _LogNorm(_Dist):
    _support = (0.0, _math.inf)
    """scipy parametrization: lognorm(s, loc=0, scale=exp(mu))."""

    def fit(self, data, *args, floc=None, fscale=None, **kw):
        """Maximum likelihood with a fixed location (scipy's floc).

        log(x - loc) is normal, so the MLE is closed form: mu is the mean
        of the logs and s their standard deviation with divisor n.
        Returns (s, loc, scale = exp(mu)) as scipy does. The free-location
        fit is refused rather than guessed.
        """
        del args, kw
        if floc is None:
            raise NotImplementedError(
                "lognorm.fit needs floc= (a fixed location); the free-location "
                "fit is not implemented in the native core")
        x = [float(v) - float(floc) for v in _flatten(data)]
        if not x or any(not v > 0 for v in x):
            raise ValueError("lognorm.fit: every observation must exceed floc")
        logs = [_math.log(v) for v in x]
        n = len(logs)
        mu = _bi.sum(logs) / n if fscale is None else _math.log(float(fscale))
        sd = _math.sqrt(_bi.sum((v - mu) ** 2 for v in logs) / n)
        return (sd, float(floc), _math.exp(mu))

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


    def sf(self, x, s, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 1.0 if v <= loc else float(norm.sf(_math.log((v - loc) / scale) / s)), x)

    def logcdf(self, x, s, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: -_math.inf if v <= loc
                          else float(norm.logcdf(_math.log((v - loc) / scale) / s)), x)

    def logsf(self, x, s, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if v <= loc
                          else float(norm.logsf(_math.log((v - loc) / scale) / s)), x)

    def logpdf(self, x, s, loc=0.0, scale=1.0):
        def one(v):
            if v <= loc:
                return -_math.inf
            lz = _math.log((v - loc) / scale)
            return -lz - _math.log(s * scale * _math.sqrt(2.0 * _math.pi)) - lz * lz / (2.0 * s * s)
        return _maybe_map(one, x)

    def ppf(self, q, s, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return float(loc)
            if p == 1.0:
                return _math.inf
            return loc + scale * _math.exp(s * float(norm.ppf(p)))
        return _maybe_map(one, q)

    def isf(self, q, s, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return _math.inf
            if p == 1.0:
                return float(loc)
            return loc + scale * _math.exp(s * float(norm.isf(p)))
        return _maybe_map(one, q)

    def mean(self, s, loc=0.0, scale=1.0):
        return loc + scale * _math.exp(0.5 * s * s)

    def var(self, s, loc=0.0, scale=1.0):
        return scale * scale * _math.expm1(s * s) * _math.exp(s * s)


class _WeibullMin(_Dist):
    _support = (0.0, _math.inf)

    def fit(self, data, *args, **kw):
        """Maximum-likelihood (c, loc, scale). With ``floc`` fixed the
        shape solves the profile equation
        sum(y^c ln y)/sum(y^c) - 1/c = mean(ln y) by Newton steps and
        the scale is (mean(y^c))^(1/c); without it the location is
        profiled out by a golden-section search below min(x)."""
        x = sorted(_flatten(data))
        n = len(x)
        if n < 2:
            raise ValueError("weibull_min.fit needs at least two observations")
        floc = kw.get("floc")
        fscale = kw.get("fscale")
        fc = kw.get("f0", kw.get("fc"))
        if len(args) >= 1 and fc is None:
            fc = args[0]

        def profile(loc):
            y = [v - loc for v in x]
            if y[0] <= 0:
                return None
            ly = [_math.log(v) for v in y]
            ml = _math.fsum(ly) / n
            if fc is not None:
                c = float(fc)
            else:
                sd = _math.sqrt(_var(ly, ddof=1)) or 1e-8
                c = _math.pi / (sd * _math.sqrt(6.0))
                for _ in range(200):
                    yc = [_math.exp(c * lv) for lv in ly]
                    s0 = _math.fsum(yc)
                    s1 = _math.fsum(v * lv for v, lv in zip(yc, ly))
                    s2 = _math.fsum(v * lv * lv for v, lv in zip(yc, ly))
                    g = s1 / s0 - 1.0 / c - ml
                    dg = (s2 * s0 - s1 * s1) / (s0 * s0) + 1.0 / (c * c)
                    step = g / dg
                    c_new = c - step
                    if c_new <= 0:
                        c_new = c / 2.0
                    if abs(c_new - c) < 1e-12 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new
            scale = float(fscale) if fscale is not None else (_math.fsum(v ** c for v in y) / n) ** (1.0 / c)
            ll = (n * _math.log(c) - n * c * _math.log(scale)
                  + (c - 1.0) * _math.fsum(ly)
                  - _math.fsum((v / scale) ** c for v in y))
            return c, scale, ll

        if floc is not None:
            r = profile(float(floc))
            if r is None:
                raise ValueError("data must exceed floc")
            return r[0], float(floc), r[1]
        span = (x[-1] - x[0]) or 1.0
        lo, hi = x[0] - 10.0 * span, x[0] - 1e-9 * span
        gr = (_math.sqrt(5.0) - 1.0) / 2.0

        def obj(loc):
            r = profile(loc)
            return -_math.inf if r is None else r[2]
        a_, b_ = lo, hi
        c1 = b_ - gr * (b_ - a_)
        c2 = a_ + gr * (b_ - a_)
        f1, f2 = obj(c1), obj(c2)
        for _ in range(200):
            if f1 < f2:
                a_, c1, f1 = c1, c2, f2
                c2 = a_ + gr * (b_ - a_)
                f2 = obj(c2)
            else:
                b_, c2, f2 = c2, c1, f1
                c1 = b_ - gr * (b_ - a_)
                f1 = obj(c1)
            if abs(b_ - a_) < 1e-10 * span:
                break
        loc = 0.5 * (a_ + b_)
        r = profile(loc)
        return r[0], loc, r[1]

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
            return 0.0 if z <= 0 else -_math.expm1(-(z ** c))
        return _maybe_map(one, x)

    def sf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 1.0 if z <= 0 else _math.exp(-(z ** c))
        return _maybe_map(one, x)

    def logpdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0:
                return -_math.inf
            if z == 0:
                return (0.0 if c == 1 else (_math.inf if c < 1 else -_math.inf)) - _math.log(scale)
            return _math.log(c) + (c - 1.0) * _math.log(z) - z ** c - _math.log(scale)
        return _maybe_map(one, x)

    def logcdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return -_math.inf
            w = z ** c
            return _math.log(-_math.expm1(-w)) if w < _math.log(2.0) else _math.log1p(-_math.exp(-w))
        return _maybe_map(one, x)

    def logsf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if v <= loc else -(((v - loc) / scale) ** c), x)

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * (-_math.log1p(-p)) ** (1.0 / c) if p < 1.0 else _math.inf
        return _maybe_map(one, q)

    def isf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * (-_math.log(p)) ** (1.0 / c) if p > 0.0 else _math.inf
        return _maybe_map(one, q)

    def mean(self, c, loc=0.0, scale=1.0):
        return loc + scale * _math.gamma(1.0 + 1.0 / c)

    def var(self, c, loc=0.0, scale=1.0):
        g1 = _math.gamma(1.0 + 1.0 / c)
        return scale * scale * (_math.gamma(1.0 + 2.0 / c) - g1 * g1)


class _NBinom(_Dist):
    _support = (0.0, _math.inf)
    _discrete = True
    def ppf(self, q, n, p):
        # walk the cdf; the mean n(1-p)/p bounds how far a quantile can sit
        kmax = int(20 * (n * (1.0 - p) / p + 1.0) + 50)
        return _maybe_map(lambda v: _ppf_discrete(
            lambda kk: self.cdf(kk, n, p), v, 0, kmax), q)

    def pmf(self, k, n, p):
        def one(kk):
            kk = int(kk)
            return _math.exp(_math.lgamma(kk + n) - _math.lgamma(n)
                             - _math.lgamma(kk + 1)
                             + n * _math.log(p)
                             + kk * _math.log1p(-p))
        return _maybe_map(one, k)

    pdf = pmf

    def logpmf(self, k, n, p):
        # scipy.stats.nbinom.logpmf, in log space so a tiny mass is not
        # lost to exp underflow, broadcasting k, n and p as scipy does
        # (a regression passes one p per observation)
        def one(kk, nn, pp):
            if kk != kk or nn != nn or pp != pp:
                return _math.nan
            if kk < 0 or kk != _math.floor(kk):
                return -_math.inf
            if pp == 1.0:
                return 0.0 if kk == 0 else -_math.inf
            return (_math.lgamma(kk + nn) - _math.lgamma(nn)
                    - _math.lgamma(kk + 1.0) + nn * _math.log(pp)
                    + kk * _math.log1p(-pp))
        from . import _array_core as _ac2
        if all(_ac2.ndim(v) == 0 for v in (k, n, p)):
            return one(float(k), float(n), float(p))
        kb, nb, pb = _ac2.broadcast_arrays(k, n, p)
        return _ac2.marr([one(a, b, c) for a, b, c in
                          zip(kb.ravel().tolist(), nb.ravel().tolist(),
                              pb.ravel().tolist())]).reshape(kb.shape)

    def cdf(self, k, n, p):
        def one(kk):
            return _betainc(n, int(kk) + 1, p)
        return _maybe_map(one, k)

    def sf(self, k, n, p):
        return 1.0 - self.cdf(k, n, p)


def _ppf_discrete(cdf_at, q, kmin, kmax):
    """Smallest integer k in [kmin, kmax] with cdf(k) >= q."""
    if not 0.0 <= q <= 1.0:
        return _math.nan
    if q == 0.0:
        return float(kmin)
    k = kmin
    while k < kmax and cdf_at(k) < q * (1.0 - 1e-12):
        k += 1
    return float(k)


class _Geom(_Dist):
    _support = (1.0, _math.inf)
    _discrete = True
    def ppf(self, q, p):
        # support k >= 1, as scipy: ceil(log(1 - q) / log(1 - p))
        def one(v):
            if not 0.0 <= v <= 1.0:
                return _math.nan
            if v == 0.0:
                return 1.0
            if v == 1.0:
                return _math.inf
            return _bi.max(1.0, float(_math.ceil(_math.log1p(-v) / _math.log1p(-p))))
        return _maybe_map(one, q)

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
    _discrete = True
    def _bounds(self, M=None, n=None, N=None):
        if None in (M, n, N):
            return (0.0, _math.inf)
        return (float(max(0, N - (M - n))), float(min(n, N)))

    def ppf(self, q, M, n, N):
        kmin = _bi.max(0, N - (M - n))
        kmax = _bi.min(n, N)
        return _maybe_map(lambda v: _ppf_discrete(
            lambda kk: self.cdf(kk, M, n, N), v, kmin, kmax), q)

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
        def one(kk):
            kk = int(_math.floor(kk))
            top = int(_bi.min(n, N))
            if kk >= top:
                return 0.0
            return _math.fsum(_scalar(self.pmf(x, M, n, N)) for x in range(_bi.max(kk + 1, 0), top + 1))
        return _maybe_map(one, k)


class _GenExtreme(_Dist):
    """scipy genextreme: c > 0 = reversed-Weibull tail, c=0 Gumbel."""

    def _bounds(self, c, loc=0.0, scale=1.0):
        c, loc, scale = float(c), float(loc), float(scale)
        if c > 0:
            return (-_math.inf, loc + scale / c)
        if c < 0:
            return (loc + scale / c, _math.inf)
        return (-_math.inf, _math.inf)

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

    @staticmethod
    def _nll(data, c, loc, scale):
        """-log L with t = 1 - c z computed as log1p(-c z) so that c -> 0
        joins the Gumbel limit smoothly."""
        if scale <= 0:
            return _math.inf
        tot = len(data) * _math.log(scale)
        for v in data:
            z = (v - loc) / scale
            if _bi.abs(c) < 1e-10:
                tot += z + _math.exp(-z)
                continue
            if c * z >= 1.0:
                return _math.inf
            lt = _math.log1p(-c * z)
            tot -= (1.0 / c - 1.0) * lt - _math.exp(lt / c)
        return tot

    def fit(self, data, *args, **kw):
        """Maximum likelihood (c, loc, scale) in scipy's parametrisation.

        Started from Hosking, Wallis & Wood's (1985) probability-weighted
        moment estimates and refined by Nelder-Mead on (c, loc, log
        scale). scipy's own fit stops at fmin's default 1e-4
        tolerances; this one runs to 1e-12, so it can sit slightly
        above scipy on the likelihood, never below.
        """
        del args, kw
        x = sorted(float(v) for v in _flatten(data))
        n = len(x)
        if n < 3:
            raise ValueError("genextreme.fit needs at least 3 observations")
        b0 = _math.fsum(x) / n
        b1 = _math.fsum((i / (n - 1.0)) * x[i] for i in range(n)) / n
        b2 = _math.fsum((i * (i - 1.0)) / ((n - 1.0) * (n - 2.0)) * x[i]
                        for i in range(n)) / n
        den = 3.0 * b2 - b0
        cc = ((2.0 * b1 - b0) / den if den != 0 else 0.0) \
            - _math.log(2.0) / _math.log(3.0)
        k = 7.8590 * cc + 2.9554 * cc * cc
        if _bi.abs(k) < 1e-6:
            alpha = (2.0 * b1 - b0) / _math.log(2.0)
            xi = b0 - 0.5772156649015329 * alpha
        else:
            g = _math.gamma(1.0 + k)
            alpha = (2.0 * b1 - b0) * k / (g * (1.0 - 2.0 ** (-k)))
            xi = b0 + alpha * (g - 1.0) / k
        if not alpha > 0:
            alpha = _math.sqrt(_math.fsum((v - b0) ** 2 for v in x) / n) or 1.0
        from . import _sci_core as _sc

        def obj(th):
            th = list(th)
            return self._nll(x, th[0], th[1], _math.exp(th[2]))
        best = None
        start = [k, xi, _math.log(alpha)]
        for _ in range(3):              # restart until the simplex settles
            r = _sc.minimize(obj, start, method="Nelder-Mead",
                             options={"xatol": 1e-12, "fatol": 1e-14,
                                      "maxiter": 20000, "maxfev": 40000})
            cand = list(r.x)
            if best is None or obj(cand) < obj(best) - 1e-15:
                best = cand
                start = cand
            else:
                break
        return float(best[0]), float(best[1]), float(_math.exp(best[2]))

    def _z_logcdf(self, z, c):
        """log F(z) = -(1 - c z)^(1/c) (scipy's sign of c); Gumbel at c = 0."""
        if c == 0.0:
            return -_math.exp(-z)
        t = 1.0 - c * z
        if t <= 0.0:
            return 0.0 if c > 0 else -_math.inf
        return -_math.exp(_math.log(t) / c)

    def logcdf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: self._z_logcdf((v - loc) / scale, c), x)

    def sf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: -_math.expm1(self._z_logcdf((v - loc) / scale, c)), x)

    def logsf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            lc = self._z_logcdf((v - loc) / scale, c)
            if lc == -_math.inf:
                return 0.0
            return _math.log(-_math.expm1(lc)) if lc > -_math.log(2.0) else _math.log1p(-_math.exp(lc))
        return _maybe_map(one, x)

    def _z_from_mlogp(self, m, c):
        # F(z) = exp(-m): z = (1 - m^c)/c = -expm1(c log m)/c
        return -_math.log(m) if c == 0.0 else -_math.expm1(c * _math.log(m)) / c

    def isf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return loc + scale * (_math.inf if c <= 0 else 1.0 / c)
            if p == 1.0:
                return loc + scale * (-_math.inf if c >= 0 else 1.0 / c)
            return loc + scale * self._z_from_mlogp(-_math.log1p(-p), c)
        return _maybe_map(one, q)

    def mean(self, c, loc=0.0, scale=1.0):
        if c <= -1.0:
            return _math.inf
        if c == 0.0:
            return loc + scale * 0.5772156649015329
        return loc + scale * (1.0 - _math.gamma(1.0 + c)) / c

    def var(self, c, loc=0.0, scale=1.0):
        if c <= -0.5:
            return _math.inf
        if c == 0.0:
            return scale * scale * _math.pi ** 2 / 6.0
        g1 = _math.gamma(1.0 + c)
        return scale * scale * (_math.gamma(1.0 + 2.0 * c) - g1 * g1) / (c * c)

    def logpdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if c == 0.0:
                return -z - _math.exp(-z) - _math.log(scale)
            t = 1.0 - c * z
            if t <= 0.0:
                return -_math.inf
            lt = _math.log(t)
            return (1.0 / c - 1.0) * lt - _math.exp(lt / c) - _math.log(scale)
        return _maybe_map(one, x)


def _gauss_legendre(n):
    """Nodes and weights on [-1, 1], by Newton iteration on P_n."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        z = _math.cos(_math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, 0.0
            for j in range(1, n + 1):
                p2 = p1
                p1 = p0
                p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j
            dp = n * (z * p0 - p1) / (z * z - 1.0)
            dz = p0 / dp
            z -= dz
            if abs(dz) < 1e-15:
                break
        nodes.append(z)
        weights.append(2.0 / ((1.0 - z * z) * dp * dp))
    return nodes, weights


class _MultivariateNormal:
    def cdf(self, x, mean=None, cov=None):
        r"""The normal CDF, exact for one and two dimensions.

        Only `pdf`/`logpdf` existed, so the Gaussian copula could not be
        evaluated at all. Two dimensions is what a copula needs and is
        the case with a clean closed form -- the tetrachoric series of
        Sheppard, via

        .. math::

           \Phi_2(h, k; \rho) = \Phi(h)\Phi(k) + \frac{1}{2\pi}
             \int_0^{\rho} \frac{1}{\sqrt{1-t^2}}
             \exp\!\left(-\frac{h^2 - 2thk + k^2}{2(1-t^2)}\right) dt,

        which is smooth on the whole path and integrates to machine
        precision with Gauss-Legendre. Three or more dimensions needs
        Genz's method -- Genz, A. (1992) "Numerical computation of
    multivariate normal probabilities", *Journal of Computational and
    Graphical Statistics* 1(2), 141-149,
    doi:10.1080/10618600.1992.10477010 -- and is refused rather than
    approximated silently.
        """
        xv = [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]
        d = len(xv)
        mu = ([0.0] * d if mean is None else
              [float(v) for v in (mean.tolist()
                                  if hasattr(mean, "tolist") else mean)])
        if cov is None:
            cm = [[1.0 if i == j else 0.0 for j in range(d)]
                  for i in range(d)]
        else:
            cm = cov.tolist() if hasattr(cov, "tolist") else cov
            cm = [[float(v) for v in r] for r in cm]
        if len(mu) != d or len(cm) != d:
            raise ValueError("multivariate_normal.cdf: x, mean and cov "
                             "disagree on dimension")
        z = [(xv[i] - mu[i]) / _math.sqrt(cm[i][i]) for i in range(d)]
        if d == 1:
            return norm.cdf(z[0])
        if d != 2:
            raise NotImplementedError(
                "multivariate_normal.cdf: only 1 and 2 dimensions are "
                f"implemented exactly; {d} needs Genz's method and is not "
                "approximated here")
        rho = cm[0][1] / _math.sqrt(cm[0][0] * cm[1][1])
        if rho <= -1.0 or rho >= 1.0:
            raise ValueError(f"multivariate_normal.cdf: correlation {rho:g} is "
                             "outside (-1, 1)")
        h, k = z[0], z[1]
        base = norm.cdf(h) * norm.cdf(k)
        if rho == 0.0:
            return base
        # 64-node Gauss-Legendre on [0, rho]
        n = 64
        nodes, weights = _gauss_legendre(n)
        half = 0.5 * rho
        total = 0.0
        for a in range(n):
            tt = half * (nodes[a] + 1.0)
            om = 1.0 - tt * tt
            total += weights[a] * _math.exp(
                -(h * h - 2.0 * tt * h * k + k * k) / (2.0 * om)
            ) / _math.sqrt(om)
        return base + half * total / (2.0 * _math.pi)

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


class _SomersDResult(tuple):
    """(statistic, pvalue) with .statistic, .pvalue and .table, as
    scipy.stats.somersd returns."""

    def __new__(cls, statistic, pvalue, table):
        obj = super().__new__(cls, (statistic, pvalue))
        obj.statistic, obj.pvalue, obj.table = statistic, pvalue, table
        obj.correlation = statistic
        return obj


def _somers_table(x, y):
    """Contingency table of x (rows, sorted levels) by y (columns)."""
    xv, yv = _flatten(x), _flatten(y)
    if len(xv) != len(yv):
        raise ValueError("x and y must have the same length")
    rows = sorted(set(xv))
    cols = sorted(set(yv))
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: j for j, v in enumerate(cols)}
    A = [[0] * len(cols) for _ in rows]
    for a, b in zip(xv, yv):
        A[ri[a]][ci[b]] += 1
    return A


def somersd(x, y=None, alternative="two-sided"):
    """Somers' D(Y|X) and its asymptotic test, as scipy.stats.somersd.

    ``x`` and ``y`` are paired ordinal samples (x is the independent
    variable), or ``x`` alone is a contingency table with the
    independent variable on the rows. With P and Q twice the concordant
    and discordant pair counts, D = (P - Q) / (N^2 - sum_i R_i^2), and
    the test statistic is Z = (P - Q) / sqrt(4 S), with
    S = sum_ij A_ij (A_ij^+ - A_ij^-)^2 - (P - Q)^2 / N (the
    Goodman-Kruskal asymptotic variance used by scipy).

    The previous version tested with Kendall's tau-a normal
    approximation, which ignores ties and so gave a different p-value
    for almost every ordinal sample, and returned no table.
    """
    if y is None:
        A = [[int(v) for v in row] for row in
             (x.tolist() if hasattr(x, "tolist") else x)]
    else:
        A = _somers_table(x, y)
    m = len(A)
    n = len(A[0]) if m else 0
    if m <= 1 or n <= 1:
        return _SomersDResult(0.0, 1.0, A)

    def block(r0, r1, c0, c1):
        tot = 0
        for i in range(r0, r1):
            for j in range(c0, c1):
                tot += A[i][j]
        return tot

    P = Q = 0
    agg = 0
    Aplus = [[0] * n for _ in range(m)]
    Aminus = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ap = block(0, i, 0, j) + block(i + 1, m, j + 1, n)
            am = block(i + 1, m, 0, j) + block(0, i, j + 1, n)
            Aplus[i][j], Aminus[i][j] = ap, am
            P += A[i][j] * ap
            Q += A[i][j] * am
            agg += A[i][j] * (ap - am) ** 2
    N = _bi.sum(_bi.sum(r) for r in A)
    sri2 = _bi.sum(_bi.sum(r) ** 2 for r in A)
    denom = N * N - sri2
    d = (P - Q) / denom if denom else _math.nan
    S = agg - (P - Q) ** 2 / N
    z = (_math.inf if P != Q else _math.nan) if S <= 0 else (P - Q) / _math.sqrt(4.0 * S)
    if _math.isnan(z):
        p = _math.nan
    elif alternative == "two-sided":
        p = _bi.min(1.0, 2.0 * norm.sf(abs(z)))
    elif alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or "
                         "'greater'")
    return _SomersDResult(float(d), float(p), A)


class _TheilslopesResult(tuple):
    """(slope, intercept, low_slope, high_slope) with attribute access."""

    def __new__(cls, slope, intercept, low_slope, high_slope):
        obj = super().__new__(cls, (slope, intercept, low_slope, high_slope))
        obj.slope, obj.intercept = slope, intercept
        obj.low_slope, obj.high_slope = low_slope, high_slope
        return obj


def theilslopes(y, x=None, alpha=0.95, method="separate"):
    del method                      # intercept: median(y) - slope * median(x)
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
    # Sen (1968) eq. 2.6 confidence limits on the slope, as scipy: the
    # rank positions of the ordered pairwise slopes at z * sigma
    if alpha > 0.5:
        alpha = 1.0 - alpha
    z = _norm_ppf(alpha / 2.0)
    nt = m
    ny = n

    def _reps(v):
        c = {}
        for u in v:
            c[u] = c.get(u, 0) + 1
        return [k for k in c.values() if k > 1]
    sigsq = (ny * (ny - 1) * (2 * ny + 5)
             - _math.fsum(k * (k - 1) * (2 * k + 5) for k in _reps(xv))
             - _math.fsum(k * (k - 1) * (2 * k + 5) for k in _reps(yv))) / 18.0
    try:
        sigma = _math.sqrt(sigsq)
        ru = _bi.min(int(round((nt - z * sigma) / 2.0)), len(slopes) - 1)
        rl = _bi.max(int(round((nt + z * sigma) / 2.0)) - 1, 0)
        low, high = slopes[rl], slopes[ru]
    except (ValueError, IndexError):
        low, high = _math.nan, _math.nan
    return _TheilslopesResult(med, inter, low, high)


def ranksums(x, y, alternative="two-sided"):
    """scipy.stats.ranksums: the Wilcoxon rank-sum z (no tie or
    continuity correction), with scipy's one-sided alternatives."""
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    ranks = rankdata(xv + yv)
    r1 = _math.fsum(ranks[:n1])
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (r1 - expected) / _math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if alternative == "two-sided":
        p = 2.0 * norm.sf(abs(z))
    elif alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    return _TestResult(z, float(p))


def _typed_table(_ac, tt):
    """A contingency table as an int-typed array, as scipy returns."""
    return _ac._typed(_ac.marr([[float(v) for v in row] for row in tt]), int)


class _MedianTestResult(tuple):
    """scipy's (statistic, pvalue, median, table) result."""

    def __new__(cls, statistic, pvalue, median, table):
        obj = super().__new__(cls, (statistic, pvalue, median, table))
        obj.statistic = statistic
        obj.pvalue = pvalue
        obj.median = median
        obj.table = table
        return obj


def median_test(*samples, ties="below", correction=True, lambda_=1,
                nan_policy="propagate"):
    """Mood's median test.

    ``ties`` says where observations exactly equal to the grand median
    go: "below" (scipy's default), "above", or "ignore" (dropped, which
    changes the totals).
    """
    if ties not in ("below", "above", "ignore"):
        raise ValueError("ties must be 'below', 'above' or 'ignore'")
    del nan_policy
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
        if ties == "below":
            below = sum(1 for u in v if u <= grand)
        elif ties == "above":
            below = sum(1 for u in v if u < grand)
            above = len(v) - below
        else:
            below = sum(1 for u in v if u < grand)
        table.append([above, below])
    tt = [[table[i][j] for i in range(len(samples))]
          for j in range(2)]
    res = chi2_contingency(tt, correction=correction, lambda_=lambda_)
    from . import _array_core as _ac
    return _MedianTestResult(res.statistic, res.pvalue, grand,
                             _typed_table(_ac, tt))


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


def _kolmogn_mtw(n, d):
    """Pr(D_n <= d), exact, by Marsaglia, Tsang & Wang (2003), "Evaluating
    Kolmogorov's distribution", J. Stat. Soft. 8(18): the (2k-1)-square
    matrix power with the scaling of the original C code."""
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    H = [[1.0 if i - j + 1 >= 0 else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        H[i][0] -= h ** (i + 1)
        H[m - 1][i] -= h ** (m - i)
    H[m - 1][0] += (2 * h - 1) ** m if 2 * h - 1 > 0 else 0.0
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 2):
                    H[i][j] /= g

    def matmul(A, B):
        return [[_math.fsum(A[i][t] * B[t][j] for t in range(m))
                 for j in range(m)] for i in range(m)]

    def rescale(A, e):
        if A[k - 1][k - 1] > 1e140:
            return [[v * 1e-140 for v in row] for row in A], e + 140
        return A, e
    # Q = H ** n by repeated squaring, exponent tracked in eQ
    Q, eQ = None, 0
    P, eP = H, 0
    nn = n
    while nn:
        if nn & 1:
            Q, eQ = (P, eP) if Q is None else rescale(matmul(Q, P), eQ + eP)
        nn >>= 1
        if nn:
            P, eP = rescale(matmul(P, P), 2 * eP)
    s = Q[k - 1][k - 1]
    for i in range(1, n + 1):
        s = s * i / n
        if s < 1e-140:
            s *= 1e140
            eQ -= 140
    return _bi.max(0.0, _bi.min(1.0, s * 10.0 ** eQ))


def _kstwo_cdf(d, n):
    d, n = float(d), int(n)
    if d <= 0.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    # exact wherever the (2k-1)-square matrix stays small -- its size is
    # set by k = floor(n d) + 1, not by n, and the powering is by
    # squaring -- the asymptotic series with Stephens correction beyond
    if int(n * d) + 1 <= 64:
        return _kolmogn_mtw(n, d)
    return 1.0 - _ks_sf(d, n)


class _KSTwo:
    """Two-sided finite-n Kolmogorov distribution: exact (Marsaglia-Tsang-
    Wang) for small n, asymptotic + Stephens correction beyond."""

    @staticmethod
    def sf(d, n):
        return 1.0 - _kstwo_cdf(d, n)

    @staticmethod
    def cdf(d, n):
        return _kstwo_cdf(d, n)

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
    _support = (0.0, _math.inf)

    def mean(self, loc=0.0, scale=1.0):
        return _math.inf

    def var(self, loc=0.0, scale=1.0):
        return _math.inf

    def std(self, loc=0.0, scale=1.0):
        return _math.inf

    def moment(self, order, loc=0.0, scale=1.0):
        return _math.inf
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
    def mean(self, b, loc=0.0, scale=1.0):
        return loc + scale * b / (b - 1.0) if b > 1 else _math.inf

    def var(self, b, loc=0.0, scale=1.0):
        return scale * scale * b / ((b - 1.0) ** 2 * (b - 2.0)) if b > 2 else _math.inf

    def std(self, b, loc=0.0, scale=1.0):
        return _math.sqrt(self.var(b, loc, scale))

    _support = (1.0, _math.inf)
    def pdf(self, x, b, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return 0.0 if z < 1.0 else b / (z ** (b + 1.0)) / scale
        return _maybe_map(one, x)


    def sf(self, x, b, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 1.0 if (v - loc) / scale <= 1.0 else ((v - loc) / scale) ** (-b), x)

    def cdf(self, x, b, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if (v - loc) / scale <= 1.0
                          else -_math.expm1(-b * _math.log((v - loc) / scale)), x)

    def logsf(self, x, b, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if (v - loc) / scale <= 1.0 else -b * _math.log((v - loc) / scale), x)

    def ppf(self, q, b, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * _math.exp(-_math.log1p(-p) / b) if p < 1.0 else _math.inf
        return _maybe_map(one, q)

    def isf(self, q, b, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * p ** (-1.0 / b) if p > 0.0 else _math.inf
        return _maybe_map(one, q)


class _GenPareto(_Dist):
    _support = (0.0, _math.inf)
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


    def _z_logsf(self, z, c):
        """log P(Z > z) for the standard generalised Pareto."""
        if z <= 0.0:
            return 0.0
        if c == 0.0:
            return -z
        if c < 0.0 and z >= -1.0 / c:
            return -_math.inf
        return -_math.log1p(c * z) / c

    def logsf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: self._z_logsf((v - loc) / scale, c), x)

    def sf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: _math.exp(self._z_logsf((v - loc) / scale, c)), x)

    def cdf(self, x, c, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: -_math.expm1(self._z_logsf((v - loc) / scale, c)), x)

    def logcdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            ls = self._z_logsf((v - loc) / scale, c)
            if ls == 0.0:
                return -_math.inf
            return _math.log(-_math.expm1(ls)) if ls > -_math.log(2.0) else _math.log1p(-_math.exp(ls))
        return _maybe_map(one, x)

    def _z_isf(self, p, c):
        # solve logsf(z) = log p
        lp = _math.log(p)
        return -lp if c == 0.0 else _math.expm1(-c * lp) / c

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 1.0:
                return loc + scale * (_math.inf if c >= 0 else -1.0 / c)
            lq = _math.log1p(-p)
            return loc + scale * (-lq if c == 0.0 else _math.expm1(-c * lq) / c)
        return _maybe_map(one, q)

    def isf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return loc + scale * (_math.inf if c >= 0 else -1.0 / c)
            return loc + scale * self._z_isf(p, c)
        return _maybe_map(one, q)

    def mean(self, c, loc=0.0, scale=1.0):
        return loc + scale / (1.0 - c) if c < 1.0 else _math.inf

    def var(self, c, loc=0.0, scale=1.0):
        return scale * scale / ((1.0 - c) ** 2 * (1.0 - 2.0 * c)) if c < 0.5 else _math.inf

    def logpdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0.0 or (c < 0.0 and z > -1.0 / c):
                return -_math.inf
            if c == 0.0:
                return -z - _math.log(scale)
            return -(1.0 / c + 1.0) * _math.log1p(c * z) - _math.log(scale)
        return _maybe_map(one, x)


@_lru_cache(maxsize=256)
def _chi2_quantile_grid(df, npts):
    """Midpoint quantiles of chi2(df), cached.

    Every chi2.ppf is itself a bisection over the incomplete gamma, so
    recomputing this grid on each noncentral cdf call dominated the
    cost; the grid depends only on (df, npts).
    """
    return tuple(chi2.ppf((i + 0.5) / npts, df) for i in range(npts))


def _nct_cdf(t, df, delta, itrmax=1000, errmax=1e-12):
    """P(T <= t) for the noncentral t: Lenth (1989) AS 243, the series
    of incomplete-beta terms R's pnt() and scipy's nctdtr use. Replaces
    a 200-point quadrature over chi-square quantiles that was off by
    about 1e-4, which is visible in a power calculation.
    """
    if t != t or df != df or delta != delta:
        return _math.nan
    if df <= 0:
        return _math.nan
    if t < 0:
        return 1.0 - _nct_cdf(-t, df, -delta, itrmax, errmax)
    if t == _math.inf:
        return 1.0
    x = t * t / (t * t + df)
    if x <= 0.0:
        return _norm_cdf(-delta)
    lam = delta * delta
    p = 0.5 * _math.exp(-0.5 * lam)
    q = _math.sqrt(2.0 / _math.pi) * p * delta
    s_ = 0.5 - p
    a = 0.5
    b = 0.5 * df
    rxb = (1.0 - x) ** b
    albeta = 0.5 * _math.log(_math.pi) + _math.lgamma(b) - _math.lgamma(0.5 + b)
    xodd = _betainc(a, b, x)
    godd = 2.0 * rxb * _math.exp(a * _math.log(x) - albeta)
    xeven = 1.0 - rxb
    geven = b * x * rxb
    tnc = p * xodd + q * xeven
    for j in range(1, itrmax + 1):
        a += 1.0
        xodd -= godd
        xeven -= geven
        godd *= x * (a + b - 1.0) / a
        geven *= x * (a + b - 0.5) / (a + 0.5)
        p *= lam / (2.0 * j)
        q *= lam / (2.0 * j + 1.0)
        s_ -= p
        tnc += p * xodd + q * xeven
        if s_ < errmax or (p * xodd + q * xeven) < errmax * tnc and j > 10:
            break
    tnc += _norm_cdf(-delta)
    return _bi.min(1.0, _bi.max(0.0, tnc))


class _NCT(_Dist):
    """Noncentral t (Lenth 1989, AS 243)."""


    # T = (Z + nc) / sqrt(V / df), V ~ chi2(df). Conditioning on V gives
    # integrals with positive integrands -- no cancellation in either tail:
    #   P(T <= x) = int chi2(v; df) Phi(x sqrt(v/df) - nc) dv
    #   P(T >  x) = int chi2(v; df) Phi(nc - x sqrt(v/df)) dv
    #   f(x)      = int chi2(v; df) phi(x sqrt(v/df) - nc) sqrt(v/df) dv
    # each summed in log space around the peak of its integrand.
    @staticmethod
    def _log_int(logg, df, x=0.0):
        # peak of the log integrand on a log grid in v, then outward; the
        # Phi factor moves the peak to v ~ df / x^2 when |x| is large
        lo = _math.log(df) - 10.0 - 2.0 * _math.log(_bi.max(1.0, _bi_abs(x)))
        n = int((_math.log(df) + 10.0 - lo) * 4.0) + 1
        grid = [_math.exp(lo + k / 4.0) for k in range(n)]
        vals = [logg(v) for v in grid]
        k = _bi.max(range(len(grid)), key=vals.__getitem__)
        vmax = grid[k]
        if vals[k] == -_math.inf:
            return -_math.inf
        up = _log_tail_integral(logg, vmax, +1)
        down = _log_tail_integral(logg, vmax, -1, bound=0.0)
        m = _bi.max(up, down)
        return m + _math.log(_math.exp(up - m) + _math.exp(down - m))

    def _log_lower(self, x, df, nc):
        return self._log_int(lambda v: -_math.inf if v <= 0 else
                             _scalar(chi2.logpdf(v, df)) + _log_ndtr(x * _math.sqrt(v / df) - nc), df, x)

    def _log_upper(self, x, df, nc):
        return self._log_int(lambda v: -_math.inf if v <= 0 else
                             _scalar(chi2.logpdf(v, df)) + _log_ndtr(nc - x * _math.sqrt(v / df)), df, x)

    def logcdf(self, x, df, nc):
        # near 0 from below, log F = log1p(-S) keeps the digits of a tiny S
        df, nc = float(df), float(nc)

        def one(v):
            lo = self._log_lower(v, df, nc)
            return lo if lo < -_math.log(2.0) else _math.log1p(-_math.exp(self._log_upper(v, df, nc)))
        return _maybe_map(one, x)

    def logsf(self, x, df, nc):
        df, nc = float(df), float(nc)

        def one(v):
            up = self._log_upper(v, df, nc)
            return up if up < -_math.log(2.0) else _math.log1p(-_math.exp(self._log_lower(v, df, nc)))
        return _maybe_map(one, x)

    def cdf(self, x, df, nc):
        df, nc = float(df), float(nc)

        def one(v):
            lo = self._log_lower(v, df, nc)
            return _math.exp(lo) if lo < -_math.log(2.0) else -_math.expm1(self._log_upper(v, df, nc))
        return _maybe_map(one, x)

    def sf(self, x, df, nc):
        df, nc = float(df), float(nc)

        def one(v):
            up = self._log_upper(v, df, nc)
            return _math.exp(up) if up < -_math.log(2.0) else -_math.expm1(self._log_lower(v, df, nc))
        return _maybe_map(one, x)

    def logpdf(self, x, df, nc):
        df, nc = float(df), float(nc)
        c = -0.5 * _math.log(2.0 * _math.pi)
        return _maybe_map(lambda xv: self._log_int(
            lambda v: -_math.inf if v <= 0 else
            _scalar(chi2.logpdf(v, df)) + c - 0.5 * (xv * _math.sqrt(v / df) - nc) ** 2
            + 0.5 * _math.log(v / df), df, xv), x)

    def pdf(self, x, df, nc):
        return _maybe_map(lambda v: _math.exp(_scalar(self.logpdf(v, df, nc))), x)

    def ppf(self, q, df, nc):
        df, nc = float(df), float(nc)

        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p in (0.0, 1.0):
                return -_math.inf if p == 0.0 else _math.inf
            return _invert(lambda v: _scalar(self.cdf(v, df, nc)), nc - 10.0, nc + 10.0, p=p,
                           sf=lambda v: _scalar(self.sf(v, df, nc)))
        return _maybe_map(one, q)

    def isf(self, q, df, nc):
        df, nc = float(df), float(nc)

        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p in (0.0, 1.0):
                return _math.inf if p == 0.0 else -_math.inf
            return _invert(lambda v: _scalar(self.cdf(v, df, nc)), nc - 10.0, nc + 10.0, q=p,
                           sf=lambda v: _scalar(self.sf(v, df, nc)))
        return _maybe_map(one, q)

    def mean(self, df, nc):
        df, nc = float(df), float(nc)
        if df <= 1.0:
            return _math.nan
        return nc * _math.sqrt(df / 2.0) * _math.exp(_math.lgamma((df - 1.0) / 2.0) - _math.lgamma(df / 2.0))

    def var(self, df, nc):
        df, nc = float(df), float(nc)
        if df <= 2.0:
            return _math.inf if df > 1.0 else _math.nan
        m = self.mean(df, nc)
        return df * (1.0 + nc * nc) / (df - 2.0) - m * m


class _NCF(_Dist):
    """Noncentral F as a Poisson(nc/2) mixture of central F: with
    U ~ chi2'(dfn, nc) = sum_j P(j) chi2(dfn + 2j), the ratio
    (U/dfn)/(V/dfd) is (dfn+2j)/dfn times a central F(dfn+2j, dfd), so
    both cdf and pdf are exact sums of central-F terms."""

    _support = (0.0, _math.inf)

    @staticmethod
    def _mix(nc, fn):
        lam = nc / 2.0
        total = 0.0
        pw = _math.exp(-lam)
        j = 0
        stop = lam + 40.0 * _math.sqrt(lam) + 40.0
        while True:
            total += pw * fn(j)
            j += 1
            pw *= lam / j
            if j > stop or (pw < 1e-17 and j > lam):
                break
        return total

    def cdf(self, x, dfn, dfd, nc):
        dfn, dfd, nc = float(dfn), float(dfd), float(nc)

        def one(v):
            if v != v:
                return _math.nan
            if v <= 0.0:
                return 0.0
            return self._mix(nc, lambda j: f.cdf(dfn * v / (dfn + 2 * j),
                                                 dfn + 2 * j, dfd))
        return _maybe_map(one, x)

    def pdf(self, x, dfn, dfd, nc):
        dfn, dfd, nc = float(dfn), float(dfd), float(nc)

        def one(v):
            if v != v:
                return _math.nan
            if v < 0.0:
                return 0.0
            return self._mix(nc, lambda j: dfn / (dfn + 2 * j)
                             * f.pdf(dfn * v / (dfn + 2 * j), dfn + 2 * j, dfd))
        return _maybe_map(one, x)

    def sf(self, x, dfn, dfd, nc):
        c = self.cdf(x, dfn, dfd, nc)
        return 1.0 - c if isinstance(c, float) else 1.0 - c

    def ppf(self, q, dfn, dfd, nc):
        def one(p):
            return _ppf_from_cdf(lambda v: self.cdf(v, dfn, dfd, nc), p, 0.0, 1e6)
        return _maybe_map(one, q)

    def mean(self, dfn, dfd, nc):
        dfn, dfd, nc = float(dfn), float(dfd), float(nc)
        if dfd <= 2.0:
            return _math.inf
        return dfd * (dfn + nc) / (dfn * (dfd - 2.0))

    def var(self, dfn, dfd, nc):
        dfn, dfd, nc = float(dfn), float(dfd), float(nc)
        if dfd <= 4.0:
            return _math.inf
        return (2.0 * (dfd / dfn) ** 2
                * ((dfn + nc) ** 2 + (dfn + 2.0 * nc) * (dfd - 2.0))
                / ((dfd - 2.0) ** 2 * (dfd - 4.0)))

    def std(self, dfn, dfd, nc):
        return _math.sqrt(self.var(dfn, dfd, nc))


halfcauchy = _HalfCauchy()
pareto = _Pareto()
genpareto = _GenPareto()
nct = _NCT()
ncf = _NCF()


# ---------------------------------------------------- further distributions

def _bessel_i(v, x):
    """Modified Bessel function I_v(x) by its power series (v >= 0 or an
    integer; the integer case uses I_{-n} = I_n)."""
    if v < 0 and float(v).is_integer():
        v = -v
    x = float(x)
    term = (x / 2.0) ** v / _math.gamma(v + 1.0)
    total = term
    q = (x / 2.0) ** 2
    k = 0
    while term > 1e-17 * total or k < 5:
        k += 1
        term *= q / (k * (k + v))
        total += term
        if k > 10000:
            break
    return total


def _owens_t(h, a):
    """Owen's T(h, a) = (1/2pi) int_0^a exp(-h^2(1+x^2)/2)/(1+x^2) dx,
    by composite Simpson (a may be negative: T is odd in a)."""
    if a == 0.0:
        return 0.0
    sign = 1.0 if a > 0 else -1.0
    a = _bi.abs(a)
    npan = 2000
    hstep = a / npan

    def fx(x):
        return _math.exp(-0.5 * h * h * (1.0 + x * x)) / (1.0 + x * x)
    s = fx(0.0) + fx(a)
    for i in range(1, npan):
        s += (4.0 if i % 2 else 2.0) * fx(i * hstep)
    return sign * s * hstep / 3.0 / (2.0 * _math.pi)


def _discrete_ppf(q, pmf, lo, hi=None):
    """Smallest k >= lo with cdf(k) >= q, walking the mass function."""
    q = float(q)
    if q <= 0.0:
        return float(lo)
    k = int(lo)
    acc = 0.0
    limit = k + 10 ** 6 if hi is None else int(hi)
    while k <= limit:
        acc += pmf(k)
        if acc >= q * (1.0 - 1e-12):
            return float(k)
        k += 1
    return float(limit)


class _Rayleigh(_Dist):
    _support = (0.0, _math.inf)

    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return z * _math.exp(-0.5 * z * z) / scale if z >= 0 else 0.0
        return _maybe_map(one, x)


    def cdf(self, x, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if v <= loc else -_math.expm1(-0.5 * ((v - loc) / scale) ** 2), x)

    def sf(self, x, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 1.0 if v <= loc else _math.exp(-0.5 * ((v - loc) / scale) ** 2), x)

    def logpdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            return -_math.inf if z <= 0 else _math.log(z) - 0.5 * z * z - _math.log(scale)
        return _maybe_map(one, x)

    def logcdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            if v <= loc:
                return -_math.inf
            w = 0.5 * ((v - loc) / scale) ** 2
            return _math.log(-_math.expm1(-w)) if w < _math.log(2.0) else _math.log1p(-_math.exp(-w))
        return _maybe_map(one, x)

    def logsf(self, x, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if v <= loc else -0.5 * ((v - loc) / scale) ** 2, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * _math.sqrt(-2.0 * _math.log1p(-p)) if p < 1.0 else _math.inf
        return _maybe_map(one, q)

    def isf(self, q, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            return loc + scale * _math.sqrt(-2.0 * _math.log(p)) if p > 0.0 else _math.inf
        return _maybe_map(one, q)

    def mean(self, loc=0.0, scale=1.0):
        return loc + scale * _math.sqrt(_math.pi / 2.0)

    def var(self, loc=0.0, scale=1.0):
        return scale * scale * (4.0 - _math.pi) / 2.0


class _InvGamma(_Dist):
    _support = (0.0, _math.inf)

    def mean(self, a, loc=0.0, scale=1.0):
        return loc + scale / (a - 1.0) if a > 1 else _math.inf

    def var(self, a, loc=0.0, scale=1.0):
        return scale * scale / ((a - 1.0) ** 2 * (a - 2.0)) if a > 2 else _math.inf

    def std(self, a, loc=0.0, scale=1.0):
        return _math.sqrt(self.var(a, loc, scale))


    # X = scale / G with G ~ Gamma(a): P(X <= x) = Q(a, scale / x)
    def cdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 0.0 if v <= loc else _gammainc_q(a, scale / (v - loc)), x)

    def sf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: 1.0 if v <= loc else _gammainc_p(a, scale / (v - loc)), x)

    def logpdf(self, x, a, loc=0.0, scale=1.0):
        def one(v):
            if v <= loc:
                return -_math.inf
            y = (v - loc) / scale
            return -_math.lgamma(a) - (a + 1.0) * _math.log(y) - 1.0 / y - _math.log(scale)
        return _maybe_map(one, x)

    def pdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: _math.exp(_scalar(self.logpdf(v, a, loc, scale))), x)

    def ppf(self, q, a, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            g = _scalar(gamma.isf(p, a))
            return loc + (scale / g if g > 0 else _math.inf)
        return _maybe_map(one, q)

    def isf(self, q, a, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            g = _scalar(gamma.ppf(p, a))
            return loc + (scale / g if g > 0 else _math.inf)
        return _maybe_map(one, q)


class _Triang(_Dist):
    _support = (0.0, 1.0)

    def _bounds(self, c=None, loc=0.0, scale=1.0):
        return (loc, loc + scale)

    def pdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z < 0 or z > 1:
                return 0.0
            if z < c:
                return 2.0 * z / c / scale
            if z > c:
                return 2.0 * (1.0 - z) / (1.0 - c) / scale
            return 2.0 / scale
        return _maybe_map(one, x)

    def cdf(self, x, c, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return 0.0
            if z >= 1:
                return 1.0
            return z * z / c if z <= c else 1.0 - (1.0 - z) ** 2 / (1.0 - c)
        return _maybe_map(one, x)

    def ppf(self, q, c, loc=0.0, scale=1.0):
        def one(p):
            z = _math.sqrt(p * c) if p <= c else 1.0 - _math.sqrt((1.0 - p) * (1.0 - c))
            return loc + scale * z
        return _maybe_map(one, q)


class _Wald(_Dist):
    """Inverse Gaussian with mean 1 and shape 1 (scipy's wald)."""
    _support = (0.0, _math.inf)

    def pdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return 0.0
            return _math.exp(-(z - 1.0) ** 2 / (2.0 * z)) / _math.sqrt(2.0 * _math.pi * z ** 3) / scale
        return _maybe_map(one, x)

    def cdf(self, x, loc=0.0, scale=1.0):
        def one(v):
            z = (v - loc) / scale
            if z <= 0:
                return 0.0
            r = _math.sqrt(z)
            return _norm_cdf((z - 1.0) / r) + _math.exp(2.0) * _norm_cdf(-(z + 1.0) / r)
        return _maybe_map(one, x)

    def ppf(self, q, loc=0.0, scale=1.0):
        return _maybe_map(lambda p: loc + scale * _ppf_from_cdf(
            lambda v: _scalar(self.cdf(v)), p, 0.0, 1e6), q)


_GL30 = None


def _log_tail_integral(logpdf, x, sgn, bound=None):
    """log of the integral of exp(logpdf) from x outward (sgn = -1: down
    to -inf or ``bound``; +1: up to +inf or ``bound``), in log space so a
    tail far beyond double range of the density ratio keeps its digits.

    Gauss-Legendre (30 points) on panels that start at the local length
    scale of the log density (the e-fold length along its slope, or its
    curvature width at a peak) and double. Toward a finite ``bound`` the
    panels switch to halving the remaining distance, so an algebraic
    endpoint behaviour (a chi-square weight v^(k/2-1) at 0) converges
    geometrically instead of at the slow rate of one wide panel."""
    global _GL30
    if _GL30 is None:
        _GL30 = _gauss_legendre(30)
    nodes, weights = _GL30
    l0 = logpdf(x)
    if l0 == -_math.inf or l0 != l0:
        return l0
    # difference steps relative to |x|: a peak at v ~ 1e-40 needs steps
    # on that scale, not 1e-6
    xs = _bi_abs(x) if x != 0 else 1.0
    d = 1e-6 * xs
    slope = (logpdf(x + sgn * d) - l0) / d
    h = 1.0 / -slope if slope < 0 and slope == slope else _math.inf
    d2 = 1e-3 * xs
    if bound is not None:
        d2 = _bi.min(d2, 0.25 * _bi_abs(x - bound))
    if d2 > 0:
        lp, lm = logpdf(x + sgn * d2), logpdf(x - sgn * d2)
        curv = (lp - 2.0 * l0 + lm) / (d2 * d2)
        if curv < 0 and curv == curv:
            h = _bi.min(h, 1.0 / _math.sqrt(-curv))
    if h == _math.inf:
        h = 1.0
    h = _bi.min(_bi.max(h, 1e-12 * xs), 1e6 * _bi.max(1.0, xs))
    total, a, width = 0.0, x, h
    for panel in range(600):
        if bound is not None and width >= 0.5 * _bi_abs(a - bound):
            b = bound + 0.5 * (a - bound)          # geometric grading
            if _bi_abs(b - bound) <= 1e-300:
                b = bound
        else:
            b = a + sgn * width
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        part = 0.0
        for t, w in zip(nodes, weights):
            lv = logpdf(mid + half * t)
            if lv != -_math.inf:
                part += w * _math.exp(lv - l0)
        part *= _bi_abs(half)
        total += part
        if b == bound or (panel > 2 and part <= 1e-17 * total):
            break
        a, width = b, width * 2.0
    return l0 + _math.log(total) if total > 0 else -_math.inf


class _SkewNorm(_Dist):


    @staticmethod
    def _std_logpdf(z, a):
        return _math.log(2.0) - 0.5 * z * z - 0.5 * _math.log(2.0 * _math.pi) + _log_ndtr(a * z)

    def _log_lower(self, z, a):
        """log P(Z <= z) for the standard skew normal."""
        if a < 0:
            return self._log_upper(-z, -a)
        lp = lambda t: self._std_logpdf(t, a)  # noqa: E731
        if z <= 0.0:
            return _log_tail_integral(lp, z, -1)
        u = _log_tail_integral(lp, z, +1)
        return _math.log1p(-_math.exp(u))

    def _log_upper(self, z, a):
        if a < 0:
            return self._log_lower(-z, -a)
        lp = lambda t: self._std_logpdf(t, a)  # noqa: E731
        if z >= 0.0:
            return _log_tail_integral(lp, z, +1)
        lo = _log_tail_integral(lp, z, -1)
        return _math.log1p(-_math.exp(lo))

    def logpdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: self._std_logpdf((v - loc) / scale, a) - _math.log(scale), x)

    def pdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: _math.exp(self._std_logpdf((v - loc) / scale, a)) / scale, x)

    def logcdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: self._log_lower((v - loc) / scale, a), x)

    def logsf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: self._log_upper((v - loc) / scale, a), x)

    def cdf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: _math.exp(self._log_lower((v - loc) / scale, a)), x)

    def sf(self, x, a, loc=0.0, scale=1.0):
        return _maybe_map(lambda v: _math.exp(self._log_upper((v - loc) / scale, a)), x)

    def ppf(self, q, a, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p in (0.0, 1.0):
                return -_math.inf if p == 0.0 else _math.inf
            return loc + scale * _invert(lambda v: _math.exp(self._log_lower(v, a)), -60.0, 60.0,
                                         p=p, sf=lambda v: _math.exp(self._log_upper(v, a)))
        return _maybe_map(one, q)

    def isf(self, q, a, loc=0.0, scale=1.0):
        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p in (0.0, 1.0):
                return _math.inf if p == 0.0 else -_math.inf
            return loc + scale * _invert(lambda v: _math.exp(self._log_lower(v, a)), -60.0, 60.0,
                                         q=p, sf=lambda v: _math.exp(self._log_upper(v, a)))
        return _maybe_map(one, q)

    def mean(self, a, loc=0.0, scale=1.0):
        d = a / _math.sqrt(1.0 + a * a)
        return loc + scale * d * _math.sqrt(2.0 / _math.pi)

    def var(self, a, loc=0.0, scale=1.0):
        d = a / _math.sqrt(1.0 + a * a)
        return scale * scale * (1.0 - 2.0 * d * d / _math.pi)


class _TruncNorm(_Dist):
    def _bounds(self, a=None, b=None, loc=0.0, scale=1.0):
        if a is None:
            return self._support
        return (loc + a * scale, loc + b * scale)

    def pdf(self, x, a, b, loc=0.0, scale=1.0):
        den = _norm_cdf(b) - _norm_cdf(a)

        def one(v):
            z = (v - loc) / scale
            if z < a or z > b:
                return 0.0
            return _math.exp(-0.5 * z * z) / _math.sqrt(2.0 * _math.pi) / den / scale
        return _maybe_map(one, x)

    def cdf(self, x, a, b, loc=0.0, scale=1.0):
        fa, fb = _norm_cdf(a), _norm_cdf(b)

        def one(v):
            z = (v - loc) / scale
            if z <= a:
                return 0.0
            if z >= b:
                return 1.0
            return (_norm_cdf(z) - fa) / (fb - fa)
        return _maybe_map(one, x)

    def ppf(self, q, a, b, loc=0.0, scale=1.0):
        fa, fb = _norm_cdf(a), _norm_cdf(b)
        return _maybe_map(lambda p: loc + scale * _norm_ppf(fa + p * (fb - fa)), q)


class _VonMises(_Dist):
    _support = (-_math.pi, _math.pi)

    def pdf(self, x, kappa, loc=0.0, scale=1.0):
        norm_c = 2.0 * _math.pi * _bessel_i(0.0, kappa)

        def one(v):
            z = (v - loc) / scale
            return _math.exp(kappa * _math.cos(z)) / norm_c / scale
        return _maybe_map(one, x)

    def cdf(self, x, kappa, loc=0.0, scale=1.0):
        norm_c = 2.0 * _math.pi * _bessel_i(0.0, kappa)

        def one(v):
            z = (v - loc) / scale
            if z <= -_math.pi:
                return 0.0
            if z >= _math.pi:
                return 1.0
            npan = 2000
            h = (z + _math.pi) / npan
            s = _math.exp(kappa * _math.cos(-_math.pi)) + _math.exp(kappa * _math.cos(z))
            for i in range(1, npan):
                s += (4.0 if i % 2 else 2.0) * _math.exp(kappa * _math.cos(-_math.pi + i * h))
            return s * h / 3.0 / norm_c
        return _maybe_map(one, x)


    def mean(self, kappa, loc=0.0, scale=1.0):
        return float(loc)                    # symmetric about loc

    def ppf(self, q, kappa, loc=0.0, scale=1.0):
        return _maybe_map(lambda p: float(loc) if p == 0.5 else loc + scale * _ppf_from_cdf(
            lambda v: _scalar(self.cdf(v, kappa)), p, -_math.pi, _math.pi), q)

    def isf(self, q, kappa, loc=0.0, scale=1.0):
        # symmetric: the upper quantile mirrors the lower one about loc
        return _maybe_map(lambda p: 2.0 * loc - _scalar(self.ppf(p, kappa, loc, scale)), q)

    def var(self, kappa, loc=0.0, scale=1.0):
        """Variance of the angle on (-pi, pi] about loc (scipy's linear,
        not circular, variance): E[theta^2] by 30-point Gauss-Legendre on
        64 panels -- the density is entire, so this is exact to rounding."""
        global _GL30
        if _GL30 is None:
            _GL30 = _gauss_legendre(30)
        nodes, weights = _GL30
        w = 2.0 * _math.pi / 64.0
        total = 0.0
        for i in range(64):
            a = -_math.pi + i * w
            for t, wt in zip(nodes, weights):
                x = a + 0.5 * w * (t + 1.0)
                total += wt * x * x * _scalar(self.pdf(x, kappa))
        return scale * scale * total * 0.5 * w


class _Bernoulli(_Dist):
    _discrete = True
    _support = (0.0, 1.0)

    def pmf(self, k, p):
        return _maybe_map(lambda v: p if v == 1 else (1.0 - p if v == 0 else 0.0), k)

    def cdf(self, k, p):
        return _maybe_map(lambda v: 0.0 if v < 0 else (1.0 - p if v < 1 else 1.0), k)

    def ppf(self, q, p):
        return _maybe_map(lambda u: 0.0 if u <= 1.0 - p else 1.0, q)


class _RandInt(_Dist):
    """Uniform integers on [low, high), scipy's randint."""
    _discrete = True

    def _bounds(self, low=None, high=None):
        if low is None:
            return self._support
        return (float(low), float(high) - 1.0)

    def pmf(self, k, low, high):
        n = float(high - low)
        return _maybe_map(lambda v: 1.0 / n if low <= v < high and float(v).is_integer()
                          else 0.0, k)

    def cdf(self, k, low, high):
        n = float(high - low)
        return _maybe_map(lambda v: 0.0 if v < low else
                          (1.0 if v >= high - 1 else (_math.floor(v) - low + 1) / n), k)

    def ppf(self, q, low, high):
        n = high - low
        return _maybe_map(lambda u: float(low + _bi.min(n - 1, _bi.max(0, _math.ceil(u * n) - 1))), q)


class _BetaBinom(_Dist):
    _discrete = True

    def _bounds(self, n=None, a=None, b=None):
        return (0.0, float(n) if n is not None else _math.inf)

    def pmf(self, k, n, a, b):
        n = int(n)

        def one(v):
            kk = int(round(v))
            if kk < 0 or kk > n:
                return 0.0
            return _math.exp(_log_comb(n, kk) + _math.lgamma(kk + a) + _math.lgamma(n - kk + b)
                             - _math.lgamma(n + a + b) + _math.lgamma(a + b)
                             - _math.lgamma(a) - _math.lgamma(b))
        return _maybe_map(one, k)

    def cdf(self, k, n, a, b):
        return _maybe_map(lambda v: _math.fsum(_scalar(self.pmf(i, n, a, b))
                                               for i in range(0, int(_math.floor(v)) + 1))
                          if v >= 0 else 0.0, k)

    def ppf(self, q, n, a, b):
        return _maybe_map(lambda u: _discrete_ppf(u, lambda i: _scalar(self.pmf(i, n, a, b)),
                                                  0, n), q)

    def sf(self, k, n, a, b):
        n = int(n)
        return _maybe_map(lambda v: 1.0 if v < 0 else _math.fsum(
            _scalar(self.pmf(i, n, a, b)) for i in range(int(_math.floor(v)) + 1, n + 1)), k)


class _Zipf(_Dist):
    _discrete = True
    _support = (1.0, _math.inf)

    @staticmethod
    def _zeta(a):
        # direct sum plus the Euler-Maclaurin tail
        N = 2000
        s = _math.fsum(k ** (-a) for k in range(1, N + 1))
        return s + N ** (1.0 - a) / (a - 1.0) - 0.5 * N ** (-a) + a * N ** (-a - 1.0) / 12.0

    def pmf(self, k, a):
        z = self._zeta(a)
        return _maybe_map(lambda v: v ** (-a) / z if v >= 1 and float(v).is_integer() else 0.0, k)

    def cdf(self, k, a):
        z = self._zeta(a)
        return _maybe_map(lambda v: _math.fsum(i ** (-a) for i in range(1, int(_math.floor(v)) + 1)) / z
                          if v >= 1 else 0.0, k)

    def ppf(self, q, a):
        z = self._zeta(a)
        return _maybe_map(lambda u: _discrete_ppf(u, lambda i: i ** (-a) / z, 1), q)

    def mean(self, a):
        return self._zeta(a - 1.0) / self._zeta(a) if a > 2 else _math.inf

    def var(self, a):
        if a <= 3:
            return _math.inf
        mu = self.mean(a)
        return self._zeta(a - 2.0) / self._zeta(a) - mu * mu

    def std(self, a):
        return _math.sqrt(self.var(a))


class _Skellam(_Dist):
    _discrete = True

    def pmf(self, k, mu1, mu2):
        mu1, mu2 = float(mu1), float(mu2)

        def one(v):
            kk = int(round(v))
            return (_math.exp(-(mu1 + mu2)) * (mu1 / mu2) ** (kk / 2.0)
                    * _bessel_i(_bi.abs(kk), 2.0 * _math.sqrt(mu1 * mu2)))
        return _maybe_map(one, k)

    def _kmin(self, mu1, mu2):
        return int(_math.floor(mu1 - mu2 - 12.0 * _math.sqrt(mu1 + mu2) - 20.0))

    def cdf(self, k, mu1, mu2):
        lo = self._kmin(float(mu1), float(mu2))
        return _maybe_map(lambda v: _math.fsum(_scalar(self.pmf(i, mu1, mu2))
                                               for i in range(lo, int(_math.floor(v)) + 1)), k)

    def ppf(self, q, mu1, mu2):
        lo = self._kmin(float(mu1), float(mu2))
        return _maybe_map(lambda u: _discrete_ppf(u, lambda i: _scalar(self.pmf(i, mu1, mu2)), lo), q)

    def _bounds(self, mu1=None, mu2=None):
        return (-_math.inf, _math.inf)


class _NCX2(_Dist):
    """Noncentral chi-square as the Poisson(nc/2) mixture of central
    chi-square(df + 2j)."""
    _support = (0.0, _math.inf)

    def cdf(self, x, df, nc):
        df, nc = float(df), float(nc)
        return _maybe_map(lambda v: 0.0 if v <= 0 else _NCF._mix(
            nc, lambda j: _scalar(chi2.cdf(v, df + 2 * j))), x)


    def mean(self, df, nc):
        return float(df) + float(nc)

    def var(self, df, nc):
        return 2.0 * (float(df) + 2.0 * float(nc))

    def std(self, df, nc):
        return _math.sqrt(self.var(df, nc))

    @staticmethod
    def _log_mix(nc, logterm):
        """log sum_j Pois(j; nc/2) exp(logterm(j)), summed in log space
        outward from the largest term until terms fall below 1e-17 of it."""
        lam = nc / 2.0

        def lw(j):
            return -lam + (j * _math.log(lam) if lam > 0 else (0.0 if j == 0 else -_math.inf)) \
                - _math.lgamma(j + 1.0)
        if lam == 0.0:
            return logterm(0)
        # locate the largest term by a coarse scan
        best_j, best = 0, lw(0) + logterm(0)
        j, step = 1, 1
        while j < 10 ** 7:
            v = lw(j) + logterm(j)
            if v > best:
                best_j, best = j, v
            elif (j > lam and v < best - 50.0) or (best == -_math.inf and j > 64 * _bi.max(1.0, lam)):
                break   # past the peak, or no finite term at all
            j += step
            if j > 64:
                step = _bi.max(1, j // 64)
        terms = []
        for sgn in (-1, 1):
            k = best_j if sgn == 1 else best_j - 1
            while k >= 0:
                v = lw(k) + logterm(k)
                terms.append(v)
                if v < best - 40.0:
                    break
                k += sgn
        mx = _bi.max(terms)
        if mx == -_math.inf:
            return -_math.inf
        return mx + _math.log(_math.fsum(_math.exp(t - mx) for t in terms))

    def logpdf(self, x, df, nc):
        df, nc = float(df), float(nc)
        return _maybe_map(lambda v: -_math.inf if v <= 0 else self._log_mix(
            nc, lambda j: _scalar(chi2.logpdf(v, df + 2 * j))), x)

    def pdf(self, x, df, nc):
        return _maybe_map(lambda v: _math.exp(_scalar(self.logpdf(v, df, nc))), x)

    def logsf(self, x, df, nc):
        # chi2.sf(x, df + 2j) grows with j, so the Poisson-weighted terms
        # peak well past nc/2: sum them in log space around their maximum
        # rather than stopping when the weights get small
        df, nc = float(df), float(nc)

        def one(v):
            if v <= 0:
                return 0.0
            c = _scalar(self.cdf(v, df, nc))
            if c < 0.5:
                return _math.log1p(-c)
            return self._log_mix(nc, lambda j: _scalar(chi2.logsf(v, df + 2 * j)))
        return _maybe_map(one, x)

    def sf(self, x, df, nc):
        return _maybe_map(lambda v: _math.exp(_scalar(self.logsf(v, df, nc))), x)

    def ppf(self, q, df, nc):
        return _maybe_map(lambda p: _ppf_from_cdf(
            lambda v: _scalar(self.cdf(v, df, nc)), p, 0.0, 1e6,
            sf=lambda v: _scalar(self.sf(v, df, nc))), q)


ncx2 = _NCX2()
rayleigh = _Rayleigh()
invgamma = _InvGamma()
triang = _Triang()
wald = _Wald()
skewnorm = _SkewNorm()
truncnorm = _TruncNorm()
vonmises = _VonMises()
bernoulli = _Bernoulli()
randint = _RandInt()
betabinom = _BetaBinom()
zipf = _Zipf()
skellam = _Skellam()


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


_ANSARI_CACHE = {}


def _ansari_freqs(n1, n2):
    """Exact null frequencies of the Ansari-Bradley statistic for sample
    sizes n1, n2 without ties: the number of n1-subsets of the scores
    min(r, N+1-r), r = 1..N, at each achievable sum (AS 93 by counting).
    Returns (astart, freqs) with freqs[i] the count at sum astart + i."""
    key = (n1, n2)
    if key in _ANSARI_CACHE:
        return _ANSARI_CACHE[key]
    N = n1 + n2
    scores = [_bi.min(r, N + 1 - r) for r in range(1, N + 1)]
    smax = sum(sorted(scores)[-n1:])
    # dp[j][s]: subsets of size j with score sum s
    dp = [[0] * (smax + 1) for _ in range(n1 + 1)]
    dp[0][0] = 1
    for sc in scores:
        for j in range(n1, 0, -1):
            prev = dp[j - 1]
            cur = dp[j]
            for s in range(smax, sc - 1, -1):
                c = prev[s - sc]
                if c:
                    cur[s] += c
    astart = sum(sorted(scores)[:n1])
    freqs = dp[n1][astart:smax + 1]
    _ANSARI_CACHE[key] = (astart, freqs)
    return astart, freqs


def ansari(x, y, alternative="two-sided", method="auto"):
    """Ansari-Bradley test. ``method='auto'`` is exact (the null
    distribution enumerated) when both samples are under 55 and there
    are no ties, the normal approximation otherwise, as in scipy."""
    xv, yv = _flatten(x), _flatten(y)
    n1, n2 = len(xv), len(yv)
    n = n1 + n2
    ranks = rankdata(xv + yv)
    # Ansari-Bradley scores: min(r, N+1-r)
    scores = [_bi.min(r, n + 1.0 - r) for r in ranks]
    ab = _math.fsum(scores[:n1])
    ties = len(set(xv + yv)) < n
    if method == "auto":
        method = "exact" if (n1 < 55 and n2 < 55 and not ties) else "asymptotic"
    if method == "exact":
        astart, freqs = _ansari_freqs(n1, n2)
        total = float(sum(freqs))
        # cdf rounds the index up, sf down: the tie-free null is an
        # approximation under ties, and this avoids a Type I overshoot
        ic = int(_math.ceil(ab - astart))
        i_f = int(_math.floor(ab - astart))
        cdf = sum(freqs[:_bi.max(ic + 1, 0)]) / total
        sf = sum(freqs[_bi.max(i_f, 0):]) / total
        if alternative == "two-sided":
            pv = 2.0 * _bi.min(cdf, sf)
        elif alternative == "greater":
            pv = cdf
        else:
            pv = sf
        return _TestResult(ab, _bi.min(1.0, pv))
    if n % 2 == 0:
        mu = n1 * (n + 2.0) / 4.0
        var = n1 * n2 * (n + 2.0) * (n - 2.0) / (48.0 * (n - 1.0))
    else:
        mu = n1 * (n + 1.0) ** 2 / (4.0 * n)
        var = n1 * n2 * (n + 1.0) * (3.0 + n * n) / (48.0 * n * n)
    if ties:
        # variance under ties from the observed scores, as scipy
        fac = _math.fsum(s * s for s in scores)
        if n % 2:
            var = n1 * n2 * (16.0 * n * fac - (n + 1.0) ** 4) / (16.0 * n * n * (n - 1.0))
        else:
            var = n1 * n2 * (16.0 * fac - n * (n + 2.0) ** 2) / (16.0 * n * (n - 1.0))
    z = (ab - mu) / _math.sqrt(var)
    if alternative == "greater":
        return _TestResult(ab, norm.cdf(z))
    if alternative == "less":
        return _TestResult(ab, norm.sf(z))
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


def cramervonmises_2samp(x, y, method="auto"):
    xv, yv = sorted(_flatten(x)), sorted(_flatten(y))
    n, m = len(xv), len(yv)
    allr = rankdata(xv + yv)
    rx = allr[:n]
    ry = allr[n:]
    # Anderson, T. W. (1962) "On the distribution of the two-sample
    # Cramer-von Mises criterion", Annals of Mathematical Statistics
    # 33(3), 1148-1159, doi:10.1214/aoms/1177704477 -- computational form
    u = n * _math.fsum((rx[i] - (i + 1)) ** 2
                       for i in range(n)) \
        + m * _math.fsum((ry[j] - (j + 1)) ** 2 for j in range(m))
    nm = n + m
    t = u / (n * m * nm) - (4.0 * n * m - 1.0) / (6.0 * nm)
    if method == "auto":
        method = "exact" if _bi.max(n, m) <= 20 else "asymptotic"
    if method == "exact":
        return _TestResult(t, _cvm_2samp_exact_p(u, n, m))
    return _TestResult(t, _cvm_asymp_sf(t))


def _cvm_2samp_exact_p(u, m, n):
    """Exact p-value of the two-sample Cramer-von Mises statistic by the
    frequency recursion of Xiao, Gordon & Yakovlev (2006), J. Stat.
    Soft. 17(8), on Anderson's (1962) form U; m and n are the sample
    sizes and u the value of U."""
    m, n = int(m), int(n)
    lcm = m * n // _math.gcd(m, n)
    a = lcm // m
    b = lcm // n
    mn = m * n
    zeta = int(_math.floor(lcm ** 2 * (m + n) * (6.0 * u - mn * (4 * mn - 1))
                           / (6 * mn ** 2)))
    gs = [{0: 1}] + [{} for _ in range(m)]
    for uu in range(n + 1):
        next_gs = []
        tmp = {}
        for v, g in enumerate(gs):
            merged = dict(tmp)
            for key, c in g.items():
                merged[key] = merged.get(key, 0) + c
            res = (a * v - b * uu) ** 2
            tmp = {key + res: c for key, c in merged.items()}
            next_gs.append(tmp)
        gs = next_gs
    freq = gs[m]
    total = _math.comb(m + n, m)
    return sum(c for key, c in freq.items() if key >= zeta) / total


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


def _solve3(A, b):
    """Gaussian elimination with partial pivoting for a small system."""
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return x


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
        # scipy's rule: a least-squares quadratic in the critical values
        # through all seven log significance levels, evaluated at tn
        S = [[_math.fsum(t ** (i + j) for t in tm) for j in range(3)] for i in range(3)]
        r = [_math.fsum(ls * t ** i for t, ls in zip(tm, logsig)) for i in range(3)]
        c = _solve3(S, r)
        p = _math.exp(c[0] + c[1] * tn + c[2] * tn * tn)
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
        rng = random_state if hasattr(random_state, "random") else _ac2.random.default_rng(random_state)

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


class _Sobol:
    """scipy.stats.qmc.Sobol: Joe-Kuo (2008) direction numbers, points in
    Gray-code order, `bits` bits of resolution. Unscrambled output is
    bit-for-bit scipy's. scramble=True applies Matousek's linear matrix
    scramble and a random digital shift (the scheme scipy uses); the
    random bits come from morie's generator, so the scrambled points are
    a different draw than scipy's for the same seed, with the same
    (t, m, s)-net guarantees.
    """

    def __init__(self, d, *, scramble=True, bits=None, rng=None, seed=None,
                 optimization=None):
        from ._sobol_dirnums import POLY, VINIT
        if optimization is not None:
            raise NotImplementedError("Sobol optimization is not supported")
        d = int(d)
        if not 1 <= d <= len(POLY):
            raise ValueError(f"d must be between 1 and {len(POLY)}")
        bits = 30 if bits is None else int(bits)
        if not 1 <= bits <= 64:
            raise ValueError("bits must be between 1 and 64")
        self.d = d
        self.bits = bits
        self.scramble = bool(scramble)
        self._maxn = 2 ** bits
        v = [[0] * bits for _ in range(d)]
        for j in range(bits):
            v[0][j] = 1
        for dim in range(1, d):
            p = POLY[dim]
            m = p.bit_length() - 1
            for j in range(min(m, bits)):
                v[dim][j] = VINIT[dim][j]
            for j in range(m, bits):
                newv = v[dim][j - m]
                pow2 = 1
                for k in range(m):
                    pow2 <<= 1
                    if (p >> (m - 1 - k)) & 1:
                        newv ^= pow2 * v[dim][j - k - 1]
                v[dim][j] = newv
        for dim in range(d):
            for j in range(bits):
                v[dim][j] <<= bits - 1 - j
        shift = [0] * d
        if self.scramble:
            from . import _array_core as _ac2
            g = _ac2.random.default_rng(seed if rng is None else rng)
            for dim in range(d):
                # lower-triangular binary matrix with unit diagonal, rows
                # indexed from the most significant bit
                L = [[1 if c == r else (int(g.integers(0, 2)) if c < r else 0)
                      for c in range(bits)] for r in range(bits)]
                for j in range(bits):
                    col = v[dim][j]
                    out = 0
                    for r in range(bits):
                        acc = 0
                        for c in range(r + 1):
                            if L[r][c] and (col >> (bits - 1 - c)) & 1:
                                acc ^= 1
                        out |= acc << (bits - 1 - r)
                    v[dim][j] = out
                shift[dim] = sum(int(g.integers(0, 2)) << b for b in range(bits))
        self._v = v
        self._shift = shift
        self.reset()

    def reset(self):
        self._quasi = list(self._shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        for _ in range(int(n)):
            self._advance()
        return self

    def _advance(self):
        k = self.num_generated
        c = 0
        while (k >> c) & 1:
            c += 1
        for dim in range(self.d):
            self._quasi[dim] ^= self._v[dim][c]
        self.num_generated += 1

    def random(self, n=1):
        from . import _array_core as _ac2
        n = int(n)
        if self.num_generated + n > self._maxn:
            raise ValueError(f"at most 2**bits = {self._maxn} points can be generated")
        scale = 1.0 / self._maxn
        out = []
        for _ in range(n):
            out.append([q * scale for q in self._quasi])
            self._advance()
        return _ac2.marr(out) if out else _ac2.zeros((0, self.d))

    def random_base2(self, m):
        n = 2 ** int(m)
        total = self.num_generated + n
        if total & (total - 1):
            raise ValueError("the balance properties of Sobol points need "
                             "the total count to be a power of 2")
        return self.random(n)


class _QMC:
    LatinHypercube = _LatinHypercube
    Sobol = _Sobol


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


# ---------------------------------------------------- more scipy.stats surface

class _ModeResult(tuple):
    def __new__(cls, mode, count):
        obj = super().__new__(cls, (mode, count))
        obj.mode, obj.count = mode, count
        return obj


def mode(a, axis=0, nan_policy="propagate", keepdims=False):
    """Most common value (smallest on ties) and its count, as scipy."""
    del axis, keepdims
    v = _flatten(a)
    if nan_policy == "omit":
        v = [x for x in v if x == x]
    elif any(x != x for x in v):
        return _ModeResult(_math.nan, 0)
    if not v:
        return _ModeResult(_math.nan, 0)
    counts = {}
    for x in v:
        counts[x] = counts.get(x, 0) + 1
    best = _bi.max(counts.values())
    return _ModeResult(_bi.min(k for k, c in counts.items() if c == best), best)


def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """Cressie-Read power divergence; lambda_=1 (default) is Pearson's
    chi-square, 0 the log-likelihood ratio (G-test), -1/2 Freeman-Tukey,
    -1 modified log-likelihood, -2 Neyman."""
    del axis
    names = {"pearson": 1.0, "log-likelihood": 0.0, "freeman-tukey": -0.5,
             "mod-log-likelihood": -1.0, "neyman": -2.0, "cressie-read": 2.0 / 3.0}
    lam = 1.0 if lambda_ is None else (names[lambda_] if isinstance(lambda_, str) else float(lambda_))
    obs = [float(v) for v in _flatten(f_obs)]
    k = len(obs)
    exp_ = [float(v) for v in _flatten(f_exp)] if f_exp is not None else [_math.fsum(obs) / k] * k
    if lam == 0.0:
        stat = 2.0 * _math.fsum(o * _math.log(o / e) for o, e in zip(obs, exp_) if o > 0)
    elif lam == -1.0:
        stat = 2.0 * _math.fsum(e * _math.log(e / o) for o, e in zip(obs, exp_) if o > 0)
    else:
        stat = 2.0 / (lam * (lam + 1.0)) * _math.fsum(o * ((o / e) ** lam - 1.0)
                                                      for o, e in zip(obs, exp_))
    df = k - 1 - ddof
    return _TestResult(stat, chi2.sf(stat, df) if df > 0 else _math.nan)


def combine_pvalues(pvalues, method="fisher", weights=None):
    """Fisher's or Stouffer's combination of independent p-values."""
    ps = [float(v) for v in _flatten(pvalues)]
    k = len(ps)
    if method == "fisher":
        stat = -2.0 * _math.fsum(_math.log(v) for v in ps)
        return _TestResult(stat, chi2.sf(stat, 2 * k))
    if method == "stouffer":
        w = [1.0] * k if weights is None else [float(v) for v in _flatten(weights)]
        z = _math.fsum(wi * _norm_ppf(1.0 - v) for wi, v in zip(w, ps)) \
            / _math.sqrt(_math.fsum(wi * wi for wi in w))
        return _TestResult(z, _norm_cdf(-(z)))
    raise ValueError("method must be 'fisher' or 'stouffer'")


def entropy(pk, qk=None, base=None, axis=0):
    """Shannon entropy of a distribution (normalised), or the relative
    entropy (Kullback-Leibler divergence) against qk."""
    del axis
    p_ = [float(v) for v in _flatten(pk)]
    tot = _math.fsum(p_)
    p_ = [v / tot for v in p_]
    if qk is None:
        h = -_math.fsum(v * _math.log(v) for v in p_ if v > 0)
    else:
        q_ = [float(v) for v in _flatten(qk)]
        qt = _math.fsum(q_)
        q_ = [v / qt for v in q_]
        h = _math.fsum(v * _math.log(v / w) for v, w in zip(p_, q_) if v > 0)
    return h / _math.log(base) if base else h


# ---------------------------------------------------- scipy parity: closed-form families

def _erfc(x):
    return 2.0 * _norm_cdf(-x * _math.sqrt(2.0))


def _bessel_k(v, x):
    """Modified Bessel K_v(x), v real, x > 0, by the integral
    K_v(x) = int_0^inf exp(-x cosh t) cosh(v t) dt (composite Simpson on a
    range where the integrand has decayed)."""
    x = float(x)
    if x <= 0:
        return _math.inf
    v = _bi.abs(float(v))
    t_max = 1.0
    while x * _math.cosh(t_max) - v * t_max < 745.0 and t_max < 60.0:
        t_max += 1.0
    npan = 800
    h = t_max / npan

    def f(t):
        e = -x * _math.cosh(t) + _math.log(_math.cosh(v * t)) if v * t < 700 else -x * _math.cosh(t) + v * t - _math.log(2.0)
        return _math.exp(e) if e > -745.0 else 0.0
    s = f(0.0) + f(t_max)
    for i in range(1, npan):
        s += (4.0 if i % 2 else 2.0) * f(i * h)
    return s * h / 3.0


class _LS(_Dist):
    """A location-scale family in scipy's convention: subclasses give the
    standard-form ``_pdf``/``_cdf`` (and ``_ppf`` when closed) in the
    shape parameters; ``loc`` and ``scale`` are trailing keywords."""

    _shapes = 0

    def _split(self, args, kw):
        args = list(args)
        loc = kw.pop("loc", None)
        scale = kw.pop("scale", None)
        if loc is None:
            loc = args.pop(self._shapes) if len(args) > self._shapes else 0.0
        if scale is None:
            scale = args.pop(self._shapes) if len(args) > self._shapes else 1.0
        return tuple(float(a) for a in args[:self._shapes]), float(loc), float(scale)

    def _sup(self, *shape):
        return self._support

    def _bounds(self, *args, **kw):
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)
        return (loc + scale * lo, loc + scale * hi)

    def pdf(self, x, *args, **kw):
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z < lo or z > hi:
                return 0.0
            try:
                return self._pdf(z, *sh) / scale
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0
        return _maybe_map(one, x)
    # these bodies handle loc/scale, nan and the support themselves; the
    # generic edge wrapper would read a third shape parameter as loc
    pdf._edge_wrapped = True

    def cdf(self, x, *args, **kw):
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z <= lo:
                return 0.0
            if z >= hi:
                return 1.0
            return self._cdf_safe(z, sh, lo, hi)
        return _maybe_map(one, x)
    cdf._edge_wrapped = True

    def _cdf_safe(self, z, sh, lo, hi):
        if z <= lo:
            return 0.0
        if z >= hi:
            return 1.0
        try:
            return _bi.max(0.0, _bi.min(1.0, self._cdf(z, *sh)))
        except (ValueError, ZeroDivisionError, OverflowError):
            mid = 0.5 * (_bi.max(lo, -1e6) + _bi.min(hi, 1e6))
            return 0.0 if z < mid else 1.0

    def ppf(self, q, *args, **kw):
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return loc + scale * lo
            if p == 1.0:
                return loc + scale * hi
            if hasattr(self, "_ppf"):
                return loc + scale * self._ppf(p, *sh)
            F = lambda z: self._cdf_safe(z, sh, lo, hi)  # noqa: E731
            if type(self)._cdf is _LS._cdf:
                a, b = self._eff_range(*sh)
            else:
                a = lo + 1e-12 * _bi.max(1.0, _bi.abs(lo)) if lo > -_math.inf else -1e3
                b = hi - 1e-12 * _bi.max(1.0, _bi.abs(hi)) if hi < _math.inf else 1e3
                while a > -1e300 and F(a) > p:
                    a = a * 2.0 if a < 0 else -1.0
                while b < 1e300 and F(b) < p:
                    b = b * 2.0 if b > 0 else 1.0
            return loc + scale * _ppf_from_cdf(F, p, a, b)
        return _maybe_map(one, q)
    ppf._edge_wrapped = True

    def _eff_range(self, *sh):
        """Where the density is not negligible: stepped outward from a
        point inside the support until the density falls under 1e-16 of
        the largest value seen. Cached per shape tuple."""
        cache = self.__dict__.setdefault("_eff_cache", {})
        if sh in cache:
            return cache[sh]
        lo, hi = self._sup(*sh)
        start = 0.0 if lo < 0.0 < hi else (lo + 1.0 if hi == _math.inf else 0.5 * (lo + hi))
        if lo > -_math.inf and start <= lo:
            start = lo + 1e-3 * (1.0 if hi == _math.inf else (hi - lo))

        def pdf_at(t):
            try:
                v = self._pdf(t, *sh)
                return v if v == v else 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0
        peak = _bi.max(pdf_at(start), 1e-300)
        # right
        r = start
        step = 0.5
        while hi == _math.inf or r < hi:
            nxt = r + step
            if hi < _math.inf and nxt > hi:
                nxt = hi
            v = pdf_at(nxt)
            peak = _bi.max(peak, v)
            r = nxt
            if nxt == hi:
                break
            if v < 1e-16 * peak and step > 1.0:
                break
            step *= 1.5
            if r > 1e12:
                break
        # left
        l_ = start
        step = 0.5
        while lo == -_math.inf or l_ > lo:
            nxt = l_ - step
            if lo > -_math.inf and nxt < lo:
                nxt = lo
            v = pdf_at(nxt)
            peak = _bi.max(peak, v)
            l_ = nxt
            if nxt == lo:
                break
            if v < 1e-16 * peak and step > 1.0:
                break
            step *= 1.5
            if l_ < -1e12:
                break
        cache[sh] = (l_, r)
        return cache[sh]

    def _cdf(self, z, *sh):
        # numeric cdf for a family that gives only a density: piecewise
        # adaptive Simpson over the effective support
        l_, r = self._eff_range(*sh)
        if z <= l_:
            return 0.0
        if z >= r:
            return 1.0

        def pdf_at(t):
            try:
                v = self._pdf(t, *sh)
                return v if v == v else 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0
        total = 0.0
        a = l_
        while a < z:
            b = _bi.min(z, a + _bi.max(1.0, (r - l_) / 64.0))
            total += _adaptive_simpson(pdf_at, a, b, 1e-13, 30)
            a = b
        return total

    # Optional standard-form hooks: _sf, _isf, _logcdf, _logsf, _stdmean,
    # _stdvar. Without them the generic _Dist versions apply.
    def sf(self, x, *args, **kw):
        if not hasattr(self, "_sf"):
            return _Dist.sf(self, x, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z <= lo:
                return 1.0
            if z >= hi:
                return 0.0
            return self._sf(z, *sh)
        return _maybe_map(one, x)
    sf._edge_wrapped = True

    def isf(self, q, *args, **kw):
        if not hasattr(self, "_isf"):
            return _Dist.isf(self, q, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return loc + scale * hi
            if p == 1.0:
                return loc + scale * lo
            return loc + scale * self._isf(p, *sh)
        return _maybe_map(one, q)
    isf._edge_wrapped = True

    def logcdf(self, x, *args, **kw):
        if not hasattr(self, "_logcdf"):
            return _Dist.logcdf(self, x, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z <= lo:
                return -_math.inf
            if z >= hi:
                return 0.0
            return self._logcdf(z, *sh)
        return _maybe_map(one, x)
    logcdf._edge_wrapped = True

    def logsf(self, x, *args, **kw):
        if not hasattr(self, "_logsf"):
            return _Dist.logsf(self, x, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z <= lo:
                return 0.0
            if z >= hi:
                return -_math.inf
            return self._logsf(z, *sh)
        return _maybe_map(one, x)
    logsf._edge_wrapped = True

    def mean(self, *args, **kw):
        if not hasattr(self, "_stdmean"):
            return _Dist.mean(self, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        return loc + scale * self._stdmean(*sh)

    def var(self, *args, **kw):
        if not hasattr(self, "_stdvar"):
            return _Dist.var(self, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        return scale * scale * self._stdvar(*sh)

    def logpdf(self, x, *args, **kw):
        if not hasattr(self, "_logpdf"):
            return _Dist.logpdf(self, x, *args, **kw)
        sh, loc, scale = self._split(args, dict(kw))
        lo, hi = self._sup(*sh)

        def one(v):
            if v != v:
                return _math.nan
            z = (v - loc) / scale
            if z < lo or z > hi:
                return -_math.inf
            return self._logpdf(z, *sh) - _math.log(scale)
        return _maybe_map(one, x)
    logpdf._edge_wrapped = True


def _phi(z):
    return _math.exp(-0.5 * z * z) / _math.sqrt(2.0 * _math.pi)


class _Alpha(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, a):
        return _phi(a - 1.0 / x) / (x * x * _norm_cdf(a))

    def _cdf(self, x, a):
        return _norm_cdf(a - 1.0 / x) / _norm_cdf(a)

    def _ppf(self, q, a):
        return 1.0 / (a - _norm_ppf(q * _norm_cdf(a)))


class _Anglit(_LS):
    _support = (-_math.pi / 4.0, _math.pi / 4.0)

    def _pdf(self, x):
        return _math.cos(2.0 * x)

    def _cdf(self, x):
        return _math.sin(x + _math.pi / 4.0) ** 2

    def _ppf(self, q):
        return _math.asin(_math.sqrt(q)) - _math.pi / 4.0


class _Arcsine(_LS):
    _support = (0.0, 1.0)

    def _pdf(self, x):
        return 1.0 / (_math.pi * _math.sqrt(x * (1.0 - x))) if 0 < x < 1 else _math.inf

    def _cdf(self, x):
        return 2.0 / _math.pi * _math.asin(_math.sqrt(x))

    def _ppf(self, q):
        return _math.sin(_math.pi * q / 2.0) ** 2


class _Argus(_LS):
    _shapes = 1
    _support = (0.0, 1.0)

    @staticmethod
    def _psi(c):
        return _norm_cdf(c) - c * _phi(c) - 0.5

    def _pdf(self, x, c):
        y = 1.0 - x * x
        return c ** 3 / (_math.sqrt(2.0 * _math.pi) * self._psi(c)) * x * _math.sqrt(y) * _math.exp(-0.5 * c * c * y)

    def _cdf(self, x, c):
        return 1.0 - self._psi(c * _math.sqrt(1.0 - x * x)) / self._psi(c)


class _BetaPrime(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, a, b):
        return _math.exp((a - 1.0) * _math.log(x) - (a + b) * _math.log1p(x)
                         - (_math.lgamma(a) + _math.lgamma(b) - _math.lgamma(a + b)))

    def _cdf(self, x, a, b):
        return _scalar(beta.cdf(x / (1.0 + x), a, b))

    def _ppf(self, q, a, b):
        t = _scalar(beta.ppf(q, a, b))
        return t / (1.0 - t)


class _Bradford(_LS):
    _shapes = 1
    _support = (0.0, 1.0)

    def _pdf(self, x, c):
        return c / ((1.0 + c * x) * _math.log1p(c))

    def _cdf(self, x, c):
        return _math.log1p(c * x) / _math.log1p(c)

    def _ppf(self, q, c):
        return ((1.0 + c) ** q - 1.0) / c


def _log1p_pow(x, e):
    """log(1 + x^e) without overflowing x^e (x = 1e-300, e = -3)."""
    le = e * _math.log(x)
    if abs(le) < 700.0:
        return _math.log1p(x ** e)        # pow is correctly rounded; exp(le) is not
    return le + _math.log1p(_math.exp(-le)) if le > 0 else _math.log1p(_math.exp(le))


class _Burr(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, c, d):
        return c * d * x ** (-c - 1.0) * (1.0 + x ** (-c)) ** (-d - 1.0)

    def _cdf(self, x, c, d):
        return (1.0 + x ** (-c)) ** (-d)


    # Burr III: F = (1 + x^-c)^-d
    def _logcdf(self, x, c, d):
        return -d * _log1p_pow(x, -c)

    def _sf(self, x, c, d):
        return -_math.expm1(-d * _log1p_pow(x, -c))

    def _logsf(self, x, c, d):
        lc = -d * _log1p_pow(x, -c)
        return _math.log(-_math.expm1(lc)) if lc > -_math.log(2.0) else _math.log1p(-_math.exp(lc))

    def _ppf(self, q, c, d):
        return _math.expm1(-_math.log(q) / d) ** (-1.0 / c)

    def _isf(self, q, c, d):
        return _math.expm1(-_math.log1p(-q) / d) ** (-1.0 / c)

    def _raw(self, k, c, d):
        # E X^k = d B(d + k/c, 1 - k/c), finite for c > k
        if c <= k:
            return _math.inf
        return d * _math.exp(_lbeta(d + k / c, 1.0 - k / c))

    def _stdmean(self, c, d):
        return self._raw(1, c, d)

    def _stdvar(self, c, d):
        m = self._raw(1, c, d)
        return self._raw(2, c, d) - m * m if m < _math.inf else _math.nan


class _Burr12(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, c, d):
        return c * d * x ** (c - 1.0) * (1.0 + x ** c) ** (-d - 1.0)


    # Burr XII: S = (1 + x^c)^-d
    def _logsf(self, x, c, d):
        return -d * _math.log1p(x ** c)

    def _sf(self, x, c, d):
        return _math.exp(-d * _math.log1p(x ** c))

    def _cdf(self, x, c, d):
        return -_math.expm1(-d * _math.log1p(x ** c))

    def _logcdf(self, x, c, d):
        ls = -d * _log1p_pow(x, c)
        if ls == 0.0:
            # x^c below rounding: F = d x^c (1 - (d+1) x^c / 2 + ...)
            y = c * _math.log(x)
            return _math.log(d) + y
        return _math.log(-_math.expm1(ls)) if ls > -_math.log(2.0) else _math.log1p(-_math.exp(ls))

    def _ppf(self, q, c, d):
        return _math.expm1(-_math.log1p(-q) / d) ** (1.0 / c)

    def _isf(self, q, c, d):
        return _math.expm1(-_math.log(q) / d) ** (1.0 / c)

    def _raw(self, k, c, d):
        # E X^k = d B(d - k/c, 1 + k/c), finite for c d > k
        if c * d <= k:
            return _math.inf
        return d * _math.exp(_lbeta(d - k / c, 1.0 + k / c))

    def _stdmean(self, c, d):
        return self._raw(1, c, d)

    def _stdvar(self, c, d):
        m = self._raw(1, c, d)
        return self._raw(2, c, d) - m * m if m < _math.inf else _math.nan


class _Chi(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, df):
        return _math.exp((df - 1.0) * _math.log(x) - 0.5 * x * x - (df / 2.0 - 1.0) * _math.log(2.0) - _math.lgamma(df / 2.0))

    def _cdf(self, x, df):
        return _scalar(gamma.cdf(0.5 * x * x, df / 2.0))

    def _ppf(self, q, df):
        return _math.sqrt(2.0 * _scalar(gamma.ppf(q, df / 2.0)))


class _Cosine(_LS):
    _support = (-_math.pi, _math.pi)

    def _pdf(self, x):
        return (1.0 + _math.cos(x)) / (2.0 * _math.pi)

    def _cdf(self, x):
        return (_math.pi + x + _math.sin(x)) / (2.0 * _math.pi)


class _CrystalBall(_LS):
    _shapes = 2

    @staticmethod
    def _consts(b, m):
        A = (m / b) ** m * _math.exp(-0.5 * b * b)
        B = m / b - b
        N = 1.0 / (m / b / (m - 1.0) * _math.exp(-0.5 * b * b) + _math.sqrt(_math.pi / 2.0) * (1.0 + _math.erf(b / _math.sqrt(2.0))))
        return A, B, N

    def _pdf(self, x, b, m):
        A, B, N = self._consts(b, m)
        if x > -b:
            return N * _math.exp(-0.5 * x * x)
        return N * A * (B - x) ** (-m)

    def _cdf(self, x, b, m):
        A, B, N = self._consts(b, m)
        if x <= -b:
            return N * A * (B - x) ** (1.0 - m) / (m - 1.0)
        return N * (m / b * _math.exp(-0.5 * b * b) / (m - 1.0)
                    + _math.sqrt(_math.pi / 2.0) * (_math.erf(x / _math.sqrt(2.0)) + _math.erf(b / _math.sqrt(2.0))))


class _DGamma(_LS):
    _shapes = 1

    def _pdf(self, x, a):
        ax = _bi.abs(x)
        return 0.5 * _math.exp((a - 1.0) * _math.log(ax) - ax - _math.lgamma(a)) if ax > 0 else (0.5 if a == 1.0 else (_math.inf if a < 1 else 0.0))

    def _cdf(self, x, a):
        g = _scalar(gamma.cdf(_bi.abs(x), a))
        return 0.5 + 0.5 * g if x >= 0 else 0.5 - 0.5 * g

    def _ppf(self, q, a):
        if q >= 0.5:
            return _scalar(gamma.ppf(2.0 * q - 1.0, a))
        return -_scalar(gamma.ppf(1.0 - 2.0 * q, a))


class _DWeibull(_LS):
    _shapes = 1

    def _pdf(self, x, c):
        ax = _bi.abs(x)
        return 0.5 * c * ax ** (c - 1.0) * _math.exp(-ax ** c) if ax > 0 else (0.5 * c if c == 1.0 else (_math.inf if c < 1 else 0.0))

    def _cdf(self, x, c):
        t = 1.0 - _math.exp(-_bi.abs(x) ** c)
        return 0.5 + 0.5 * t if x >= 0 else 0.5 - 0.5 * t

    def _ppf(self, q, c):
        if q >= 0.5:
            return (-_math.log(2.0 * (1.0 - q))) ** (1.0 / c)
        return -(-_math.log(2.0 * q)) ** (1.0 / c)


class _Erlang(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, a):
        return _scalar(gamma.pdf(x, a))

    def _cdf(self, x, a):
        return _scalar(gamma.cdf(x, a))

    def _ppf(self, q, a):
        return _scalar(gamma.ppf(q, a))


class _ExponNorm(_LS):
    _shapes = 1

    def _pdf(self, x, K):
        iK = 1.0 / K
        return 0.5 * iK * _math.exp(0.5 * iK * iK - x * iK) * _erfc(-(x - iK) / _math.sqrt(2.0))

    def _cdf(self, x, K):
        iK = 1.0 / K
        return _norm_cdf(x) - _math.exp(-x * iK + 0.5 * iK * iK) * _norm_cdf(x - iK)


class _ExponPow(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, b):
        xb = x ** b
        return b * x ** (b - 1.0) * _math.exp(1.0 + xb - _math.exp(xb))

    def _cdf(self, x, b):
        return 1.0 - _math.exp(1.0 - _math.exp(x ** b))

    def _ppf(self, q, b):
        return _math.log1p(-_math.log1p(-q)) ** (1.0 / b)


class _ExponWeib(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, a, c):
        e = _math.exp(-x ** c)
        return a * c * (1.0 - e) ** (a - 1.0) * e * x ** (c - 1.0)

    def _cdf(self, x, a, c):
        return (1.0 - _math.exp(-x ** c)) ** a

    def _ppf(self, q, a, c):
        return (-_math.log1p(-q ** (1.0 / a))) ** (1.0 / c)


class _FatigueLife(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return (x + 1.0) / (2.0 * c * _math.sqrt(2.0 * _math.pi * x ** 3)) * _math.exp(-(x - 1.0) ** 2 / (2.0 * x * c * c))

    def _cdf(self, x, c):
        return _norm_cdf((_math.sqrt(x) - 1.0 / _math.sqrt(x)) / c)

    def _ppf(self, q, c):
        t = c * _norm_ppf(q)
        r = 0.5 * (t + _math.sqrt(t * t + 4.0))
        return r * r


class _Fisk(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return c * x ** (c - 1.0) / (1.0 + x ** c) ** 2

    def _cdf(self, x, c):
        return 1.0 / (1.0 + x ** (-c))


    # log-logistic = Burr III with d = 1
    def _logcdf(self, x, c):
        return -_log1p_pow(x, -c)

    def _sf(self, x, c):
        return _math.exp(-_log1p_pow(x, c))

    def _logsf(self, x, c):
        return -_log1p_pow(x, c)

    def _ppf(self, q, c):
        return _math.exp((_math.log(q) - _math.log1p(-q)) / c)

    def _isf(self, q, c):
        return _math.exp((_math.log1p(-q) - _math.log(q)) / c)

    def _stdmean(self, c):
        return (_math.pi / c) / _math.sin(_math.pi / c) if c > 1 else _math.inf

    def _stdvar(self, c):
        if c <= 2:
            return _math.inf if c > 1 else _math.nan
        b = _math.pi / c
        return 2.0 * b / _math.sin(2.0 * b) - (b / _math.sin(b)) ** 2


class _FoldCauchy(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return (1.0 / (1.0 + (x - c) ** 2) + 1.0 / (1.0 + (x + c) ** 2)) / _math.pi

    def _cdf(self, x, c):
        return (_math.atan(x - c) + _math.atan(x + c)) / _math.pi


class _FoldNorm(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return _math.sqrt(2.0 / _math.pi) * _math.cosh(c * x) * _math.exp(-0.5 * (x * x + c * c))

    def _cdf(self, x, c):
        return _norm_cdf(x - c) + _norm_cdf(x + c) - 1.0


class _GenExpon(_LS):
    _shapes = 3
    _support = (0.0, _math.inf)

    def _pdf(self, x, a, b, c):
        return (a + b * (1.0 - _math.exp(-c * x))) * _math.exp(-a * x - b * x + b / c * (1.0 - _math.exp(-c * x)))

    def _cdf(self, x, a, b, c):
        return 1.0 - _math.exp(-a * x - b * x + b / c * (1.0 - _math.exp(-c * x)))


class _GenGamma(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, a, c):
        return _math.exp(_math.log(_bi.abs(c)) + (c * a - 1.0) * _math.log(x) - x ** c - _math.lgamma(a))

    def _cdf(self, x, a, c):
        g = _scalar(gamma.cdf(x ** c, a))
        return g if c > 0 else 1.0 - g

    def _ppf(self, q, a, c):
        return _scalar(gamma.ppf(q if c > 0 else 1.0 - q, a)) ** (1.0 / c)


class _GenHalfLogistic(_LS):
    _shapes = 1

    def _sup(self, c):
        return (0.0, 1.0 / c)

    def _pdf(self, x, c):
        t = (1.0 - c * x) ** (1.0 / c)
        return 2.0 * (1.0 - c * x) ** (1.0 / c - 1.0) / (1.0 + t) ** 2

    def _cdf(self, x, c):
        t = (1.0 - c * x) ** (1.0 / c)
        return (1.0 - t) / (1.0 + t)

    def _ppf(self, q, c):
        t = (1.0 - q) / (1.0 + q)
        return (1.0 - t ** c) / c


class _GenLogistic(_LS):
    _shapes = 1

    def _pdf(self, x, c):
        return c * _math.exp(-x) / (1.0 + _math.exp(-x)) ** (c + 1.0) if x > -700 else 0.0

    def _cdf(self, x, c):
        return (1.0 + _math.exp(-x)) ** (-c) if x > -700 else 0.0

    def _ppf(self, q, c):
        return -_math.log(q ** (-1.0 / c) - 1.0)


class _GenNorm(_LS):
    _shapes = 1

    def _pdf(self, x, b):
        return b / (2.0 * _math.gamma(1.0 / b)) * _math.exp(-_bi.abs(x) ** b)

    def _cdf(self, x, b):
        g = _scalar(gamma.cdf(_bi.abs(x) ** b, 1.0 / b))
        return 0.5 + 0.5 * g if x >= 0 else 0.5 - 0.5 * g

    def _ppf(self, q, b):
        if q >= 0.5:
            return _scalar(gamma.ppf(2.0 * q - 1.0, 1.0 / b)) ** (1.0 / b)
        return -_scalar(gamma.ppf(1.0 - 2.0 * q, 1.0 / b)) ** (1.0 / b)


class _Gibrat(_LS):
    _support = (0.0, _math.inf)

    def _pdf(self, x):
        return _phi(_math.log(x)) / x

    def _cdf(self, x):
        return _norm_cdf(_math.log(x))

    def _ppf(self, q):
        return _math.exp(_norm_ppf(q))


class _Gompertz(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return c * _math.exp(x) * _math.exp(-c * (_math.exp(x) - 1.0))

    def _cdf(self, x, c):
        return 1.0 - _math.exp(-c * (_math.exp(x) - 1.0))

    def _ppf(self, q, c):
        return _math.log1p(-_math.log1p(-q) / c)


class _GumbelR(_LS):
    def _pdf(self, x):
        return _math.exp(-(x + _math.exp(-x))) if x > -700 else 0.0

    def _cdf(self, x):
        return _math.exp(-_math.exp(-x)) if x > -700 else 0.0

    def _ppf(self, q):
        return -_math.log(-_math.log(q))

    def _sf(self, x):
        return -_math.expm1(-_math.exp(-x)) if x > -700 else 1.0

    def _logcdf(self, x):
        return -_math.exp(-x) if x > -700 else -_math.inf


    def _isf(self, q):
        return -_math.log(-_math.log1p(-q))

    def _stdmean(self):
        return 0.5772156649015329          # Euler-Mascheroni

    def _stdvar(self):
        return _math.pi ** 2 / 6.0

    def _logpdf(self, x):
        return -(x + _math.exp(-x))

    def _logsf(self, x):
        w = _math.exp(-x) if x > -700 else _math.inf
        # log(1 - e^-w): log(-expm1(-w)) for small w, log1p(-e^-w) for large
        return _math.log(-_math.expm1(-w)) if w < _math.log(2.0) else _math.log1p(-_math.exp(-w))


class _GumbelL(_LS):
    def _pdf(self, x):
        return _math.exp(x - _math.exp(x)) if x < 700 else 0.0

    def _cdf(self, x):
        return -_math.expm1(-_math.exp(x)) if x < 700 else 1.0

    def _ppf(self, q):
        return _math.log(-_math.log1p(-q))


class _HalfGenNorm(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, b):
        return b / _math.gamma(1.0 / b) * _math.exp(-x ** b)

    def _cdf(self, x, b):
        return _scalar(gamma.cdf(x ** b, 1.0 / b))

    def _ppf(self, q, b):
        return _scalar(gamma.ppf(q, 1.0 / b)) ** (1.0 / b)


class _HalfLogistic(_LS):
    _support = (0.0, _math.inf)

    def _pdf(self, x):
        e = _math.exp(-x)
        return 2.0 * e / (1.0 + e) ** 2

    def _cdf(self, x):
        return _math.tanh(0.5 * x)

    def _ppf(self, q):
        return _math.log((1.0 + q) / (1.0 - q))


class _HalfNorm(_LS):
    _support = (0.0, _math.inf)

    def _pdf(self, x):
        return _math.sqrt(2.0 / _math.pi) * _math.exp(-0.5 * x * x)


    def _cdf(self, x):
        return _math.erf(x / _math.sqrt(2.0))

    def _sf(self, x):
        return _math.erfc(x / _math.sqrt(2.0))

    def _logsf(self, x):
        # log(erfc(x / sqrt 2)): near 0 erfc rounds to 1 - O(eps), so take
        # log1p(-erf) there; in the tail log 2 + log Phi(-x)
        if x < 1.0:
            return _math.log1p(-_math.erf(x / _math.sqrt(2.0)))
        return _math.log(2.0) + float(norm.logsf(x))

    def _isf(self, q):
        return float(norm.isf(0.5 * q))


    def _stdmean(self):
        return _math.sqrt(2.0 / _math.pi)

    def _stdvar(self):
        return 1.0 - 2.0 / _math.pi

    def _logpdf(self, x):
        return 0.5 * _math.log(2.0 / _math.pi) - 0.5 * x * x

    def _ppf(self, q):
        if q >= 0.5:
            return float(norm.isf(0.5 * (1.0 - q)))
        # erf(z / sqrt 2) = q: start from the linear term, Newton on erf
        z = q * _math.sqrt(_math.pi / 2.0)
        for _ in range(60):
            step = (_math.erf(z / _math.sqrt(2.0)) - q) / (_math.sqrt(2.0 / _math.pi) * _math.exp(-0.5 * z * z))
            z -= step
            if abs(step) <= 1e-17 * abs(z):
                break
        return z


class _HypSecant(_LS):
    def _pdf(self, x):
        return 1.0 / (_math.pi * _math.cosh(x)) if _bi.abs(x) < 700 else 0.0


    def _cdf(self, x):
        return 2.0 / _math.pi * _math.atan(_math.exp(x)) if x < 700 else 1.0

    def _sf(self, x):
        return 2.0 / _math.pi * _math.atan(_math.exp(-x)) if x > -700 else 1.0

    def _logsf(self, x):
        if x < 0:
            return _math.log1p(-self._sf(-x))
        return _math.log(2.0 / _math.pi) + _math.log(_math.atan(_math.exp(-x))) if x < 700 \
            else _math.log(2.0 / _math.pi) - x

    def _logcdf(self, x):
        return self._logsf(-x) if x < 0 else _math.log1p(-self._sf(x))

    def _ppf(self, q):
        return _math.log(_math.tan(0.5 * _math.pi * q))

    def _isf(self, q):
        return -_math.log(_math.tan(0.5 * _math.pi * q))

    def _logpdf(self, x):
        ax = abs(x)
        return -_math.log(_math.pi) - ax - _math.log1p(_math.exp(-2.0 * ax)) + _math.log(2.0)

    def _stdmean(self):
        return 0.0

    def _stdvar(self):
        return _math.pi ** 2 / 4.0


class _InvGauss(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, mu):
        return 1.0 / _math.sqrt(2.0 * _math.pi * x ** 3) * _math.exp(-(x - mu) ** 2 / (2.0 * x * mu * mu))

    def _cdf(self, x, mu):
        r = _math.sqrt(x)
        a = _norm_cdf((x / mu - 1.0) / r)
        b = _norm_cdf(-(x / mu + 1.0) / r)
        return a + _math.exp(2.0 / mu + _math.log(b)) if b > 0 else a


class _InvWeibull(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return c * x ** (-c - 1.0) * _math.exp(-x ** (-c))

    def _cdf(self, x, c):
        return _math.exp(-x ** (-c))


    # Frechet: F = exp(-x^-c)
    @staticmethod
    def _w(x, c):
        # x^-c, direct where it fits (pow rounds correctly, exp(-c log x) does not)
        lw = -c * _math.log(x)
        return x ** (-c) if lw < 700.0 else _math.inf

    def _logcdf(self, x, c):
        return -self._w(x, c)

    def _sf(self, x, c):
        w = self._w(x, c)
        return -_math.expm1(-w) if w < _math.inf else 1.0

    def _logsf(self, x, c):
        w = self._w(x, c)
        if w == _math.inf:
            return 0.0
        return _math.log(-_math.expm1(-w)) if w < _math.log(2.0) else _math.log1p(-_math.exp(-w))

    def _ppf(self, q, c):
        return (-_math.log(q)) ** (-1.0 / c)

    def _isf(self, q, c):
        return (-_math.log1p(-q)) ** (-1.0 / c)

    def _stdmean(self, c):
        return _math.gamma(1.0 - 1.0 / c) if c > 1 else _math.inf

    def _stdvar(self, c):
        if c <= 2:
            return _math.inf if c > 1 else _math.nan
        g1 = _math.gamma(1.0 - 1.0 / c)
        return _math.gamma(1.0 - 2.0 / c) - g1 * g1


class _IrwinHall(_LS):
    _shapes = 1

    def _sup(self, n):
        return (0.0, float(n))

    def _pdf(self, x, n):
        n = int(n)
        s = 0.0
        for k in range(0, int(_math.floor(x)) + 1):
            s += (-1) ** k * _math.comb(n, k) * (x - k) ** (n - 1)
        return s / _math.factorial(n - 1)

    def _cdf(self, x, n):
        n = int(n)
        s = 0.0
        for k in range(0, int(_math.floor(x)) + 1):
            s += (-1) ** k * _math.comb(n, k) * (x - k) ** n
        return s / _math.factorial(n)


class _JFSkewT(_LS):
    _shapes = 2

    def _pdf(self, x, a, b):
        r = _math.sqrt(a + b + x * x)
        c = _math.exp(-(a + b - 1.0) * _math.log(2.0) - (_math.lgamma(a) + _math.lgamma(b) - _math.lgamma(a + b)) - 0.5 * _math.log(a + b))
        return c * (1.0 + x / r) ** (a + 0.5) * (1.0 - x / r) ** (b + 0.5)

    def _cdf(self, x, a, b):
        return _scalar(beta.cdf(0.5 * (1.0 + x / _math.sqrt(a + b + x * x)), a, b))

    def _ppf(self, q, a, b):
        t = _scalar(beta.ppf(q, a, b))
        return (2.0 * t - 1.0) * _math.sqrt(a + b) / (2.0 * _math.sqrt(t * (1.0 - t)))


class _JohnsonSB(_LS):
    _shapes = 2
    _support = (0.0, 1.0)

    def _pdf(self, x, a, b):
        return b / (x * (1.0 - x)) * _phi(a + b * _math.log(x / (1.0 - x)))

    def _cdf(self, x, a, b):
        return _norm_cdf(a + b * _math.log(x / (1.0 - x)))


    def _z(self, x, a, b):
        return a + b * (_math.log(x) - _math.log1p(-x))

    def _sf(self, x, a, b):
        return _scalar(norm.sf(self._z(x, a, b)))

    def _logsf(self, x, a, b):
        return _scalar(norm.logsf(self._z(x, a, b)))

    def _logcdf(self, x, a, b):
        return _scalar(norm.logcdf(self._z(x, a, b)))

    def _ppf(self, q, a, b):
        return 1.0 / (1.0 + _math.exp(-(_scalar(norm.ppf(q)) - a) / b))

    def _isf(self, q, a, b):
        return 1.0 / (1.0 + _math.exp(-(_scalar(norm.isf(q)) - a) / b))


class _JohnsonSU(_LS):
    _shapes = 2

    def _pdf(self, x, a, b):
        return b / _math.sqrt(x * x + 1.0) * _phi(a + b * _math.asinh(x))

    def _cdf(self, x, a, b):
        return _norm_cdf(a + b * _math.asinh(x))


    def _z(self, x, a, b):
        return a + b * _math.asinh(x)

    def _sf(self, x, a, b):
        return _scalar(norm.sf(self._z(x, a, b)))

    def _logsf(self, x, a, b):
        return _scalar(norm.logsf(self._z(x, a, b)))

    def _logcdf(self, x, a, b):
        return _scalar(norm.logcdf(self._z(x, a, b)))

    def _ppf(self, q, a, b):
        return _math.sinh((_scalar(norm.ppf(q)) - a) / b)

    def _isf(self, q, a, b):
        return _math.sinh((_scalar(norm.isf(q)) - a) / b)

    def _stdmean(self, a, b):
        return -_math.exp(0.5 / (b * b)) * _math.sinh(a / b)

    def _stdvar(self, a, b):
        w = _math.exp(1.0 / (b * b))
        return 0.5 * (w - 1.0) * (w * _math.cosh(2.0 * a / b) + 1.0)


class _Kappa3(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, a):
        return a * (a + x ** a) ** (-(a + 1.0) / a)

    def _cdf(self, x, a):
        return x * (a + x ** a) ** (-1.0 / a)

    def _ppf(self, q, a):
        return (a * q ** a / (1.0 - q ** a)) ** (1.0 / a)


class _Kappa4(_LS):
    _shapes = 2

    def _sup(self, h, k):
        if h > 0 and k > 0:
            return ((1.0 - h ** (-k)) / k, 1.0 / k)
        if h > 0 and k == 0:
            return (_math.log(h), _math.inf)
        if h > 0 and k < 0:
            return ((1.0 - h ** (-k)) / k, _math.inf)
        if h <= 0 and k > 0:
            return (-_math.inf, 1.0 / k)
        if h <= 0 and k == 0:
            return (-_math.inf, _math.inf)
        return (1.0 / k, _math.inf)

    def _cdf(self, x, h, k):
        t = (1.0 - k * x) ** (1.0 / k) if k != 0 else _math.exp(-x)
        return (1.0 - h * t) ** (1.0 / h) if h != 0 else _math.exp(-t)

    def _pdf(self, x, h, k):
        t = (1.0 - k * x) ** (1.0 / k) if k != 0 else _math.exp(-x)
        dt = (1.0 - k * x) ** (1.0 / k - 1.0) if k != 0 else _math.exp(-x)
        return (1.0 - h * t) ** (1.0 / h - 1.0) * dt if h != 0 else _math.exp(-t) * dt

    def _ppf(self, q, h, k):
        t = (1.0 - q ** h) / h if h != 0 else -_math.log(q)
        return (1.0 - t ** k) / k if k != 0 else -_math.log(t)


class _LaplaceAsymmetric(_LS):
    _shapes = 1

    def _pdf(self, x, kappa):
        c = 1.0 / (kappa + 1.0 / kappa)
        return c * (_math.exp(-x * kappa) if x >= 0 else _math.exp(x / kappa))

    def _cdf(self, x, kappa):
        k2 = kappa * kappa
        if x < 0:
            return k2 / (1.0 + k2) * _math.exp(x / kappa)
        return 1.0 - _math.exp(-x * kappa) / (1.0 + k2)


    # kappa^2/(1+kappa^2) of the mass lies below 0
    def _sf(self, x, k):
        return _math.exp(-k * x) / (1.0 + k * k) if x >= 0 else \
            1.0 - k * k / (1.0 + k * k) * _math.exp(x / k)

    def _logsf(self, x, k):
        return -k * x - _math.log1p(k * k) if x >= 0 else _math.log1p(-k * k / (1.0 + k * k) * _math.exp(x / k))

    def _logcdf(self, x, k):
        return 2.0 * _math.log(k) - _math.log1p(k * k) + x / k if x < 0 \
            else _math.log1p(-_math.exp(-k * x) / (1.0 + k * k))

    def _logpdf(self, x, k):
        return -_math.log(k + 1.0 / k) + (-k * x if x >= 0 else x / k)

    def _ppf(self, q, k):
        t = k * k / (1.0 + k * k)
        return k * _math.log(q / t) if q < t else -(_math.log1p(-q) + _math.log1p(k * k)) / k

    def _isf(self, q, k):
        u = 1.0 / (1.0 + k * k)
        return -(_math.log(q) + _math.log1p(k * k)) / k if q < u else k * (_math.log1p(-q) - _math.log(k * k * u))

    def _stdmean(self, k):
        return 1.0 / k - k

    def _stdvar(self, k):
        return (1.0 + k ** 4) / (k * k)


class _Levy(_LS):
    _support = (0.0, _math.inf)

    def _pdf(self, x):
        return _math.exp(-0.5 / x) / (x * _math.sqrt(2.0 * _math.pi * x))

    def _cdf(self, x):
        return 2.0 * _norm_cdf(-1.0 / _math.sqrt(x))

    def _ppf(self, q):
        return 1.0 / _norm_ppf(1.0 - 0.5 * q) ** 2


class _LevyL(_LS):
    _support = (-_math.inf, 0.0)

    def _pdf(self, x):
        ax = -x
        return _math.exp(-0.5 / ax) / (ax * _math.sqrt(2.0 * _math.pi * ax))

    def _cdf(self, x):
        return 2.0 * _norm_cdf(1.0 / _math.sqrt(-x)) - 1.0

    def _ppf(self, q):
        return -1.0 / _norm_ppf(0.5 * (1.0 + q)) ** 2


class _LogGamma(_LS):
    _shapes = 1

    def _pdf(self, x, c):
        return _math.exp(c * x - _math.exp(x) - _math.lgamma(c)) if x < 700 else 0.0

    def _cdf(self, x, c):
        return _scalar(gamma.cdf(_math.exp(x), c)) if x < 700 else 1.0

    def _ppf(self, q, c):
        return _math.log(_scalar(gamma.ppf(q, c)))


class _LogLaplace(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return 0.5 * c * (x ** (c - 1.0) if x < 1.0 else x ** (-c - 1.0))

    def _cdf(self, x, c):
        return 0.5 * x ** c if x < 1.0 else 1.0 - 0.5 * x ** (-c)

    def _ppf(self, q, c):
        return (2.0 * q) ** (1.0 / c) if q < 0.5 else (2.0 * (1.0 - q)) ** (-1.0 / c)


class _Lomax(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, c):
        return c * (1.0 + x) ** (-c - 1.0)

    def _cdf(self, x, c):
        return 1.0 - (1.0 + x) ** (-c)

    def _ppf(self, q, c):
        return (1.0 - q) ** (-1.0 / c) - 1.0


class _Maxwell(_LS):
    _support = (0.0, _math.inf)

    def _pdf(self, x):
        return _math.sqrt(2.0 / _math.pi) * x * x * _math.exp(-0.5 * x * x)

    def _cdf(self, x):
        return _scalar(gamma.cdf(0.5 * x * x, 1.5))

    def _ppf(self, q):
        return _math.sqrt(2.0 * _scalar(gamma.ppf(q, 1.5)))


class _Mielke(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, k, s):
        return k * x ** (k - 1.0) / (1.0 + x ** s) ** (1.0 + k / s)

    def _cdf(self, x, k, s):
        return x ** k / (1.0 + x ** s) ** (k / s)

    def _ppf(self, q, k, s):
        t = q ** (s / k)
        return (t / (1.0 - t)) ** (1.0 / s)


class _Moyal(_LS):
    def _pdf(self, x):
        return _math.exp(-0.5 * (x + _math.exp(-x))) / _math.sqrt(2.0 * _math.pi) if x > -700 else 0.0

    def _cdf(self, x):
        return _erfc(_math.exp(-0.5 * x) / _math.sqrt(2.0)) if x > -700 else 0.0

    def _ppf(self, q):
        return -2.0 * _math.log(-_norm_ppf(0.5 * q))


class _Nakagami(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, nu):
        return _math.exp(_math.log(2.0) + nu * _math.log(nu) - _math.lgamma(nu) + (2.0 * nu - 1.0) * _math.log(x) - nu * x * x)


    # nu X^2 ~ Gamma(nu)
    def _cdf(self, x, nu):
        return _gammainc_p(nu, nu * x * x)

    def _sf(self, x, nu):
        return _gammainc_q(nu, nu * x * x)

    def _logpdf(self, x, nu):
        return (_math.log(2.0) + nu * _math.log(nu) - _math.lgamma(nu)
                + (2.0 * nu - 1.0) * _math.log(x) - nu * x * x) if x > 0 else -_math.inf

    def _ppf(self, q, nu):
        return _math.sqrt(_scalar(gamma.ppf(q, nu)) / nu)

    def _isf(self, q, nu):
        return _math.sqrt(_scalar(gamma.isf(q, nu)) / nu)

    def _stdmean(self, nu):
        return _math.exp(_math.lgamma(nu + 0.5) - _math.lgamma(nu)) / _math.sqrt(nu)

    def _stdvar(self, nu):
        m = self._stdmean(nu)
        return 1.0 - m * m


class _Pearson3(_LS):
    _shapes = 1

    def _sup(self, skew):
        if _bi.abs(skew) < 1e-10:
            return (-_math.inf, _math.inf)
        a, b, z = self._abz(skew)
        return (z, _math.inf) if b > 0 else (-_math.inf, z)

    @staticmethod
    def _abz(skew):
        b = 2.0 / skew
        a = 4.0 / (skew * skew)
        return a, b, -a / b

    def _pdf(self, x, skew):
        if _bi.abs(skew) < 1e-10:
            return _phi(x)
        a, b, z = self._abz(skew)
        y = b * (x - z)
        return _bi.abs(b) * _math.exp((a - 1.0) * _math.log(y) - y - _math.lgamma(a)) if y > 0 else 0.0

    def _cdf(self, x, skew):
        if _bi.abs(skew) < 1e-10:
            return _norm_cdf(x)
        a, b, z = self._abz(skew)
        g = _scalar(gamma.cdf(b * (x - z), a))
        return g if b > 0 else 1.0 - g

    def _ppf(self, q, skew):
        if _bi.abs(skew) < 1e-10:
            return _norm_ppf(q)
        a, b, z = self._abz(skew)
        return z + _scalar(gamma.ppf(q if b > 0 else 1.0 - q, a)) / b


class _PowerLaw(_LS):
    _shapes = 1
    _support = (0.0, 1.0)

    def _pdf(self, x, a):
        return a * x ** (a - 1.0)

    def _cdf(self, x, a):
        return x ** a

    def _ppf(self, q, a):
        return q ** (1.0 / a)

    def _sf(self, x, a):
        return -_math.expm1(a * _math.log(x)) if x > 0 else 1.0

    def _logcdf(self, x, a):
        return a * _math.log(x) if x > 0 else -_math.inf

    def _stdmean(self, a):
        return a / (a + 1.0)

    def _stdvar(self, a):
        return a / ((a + 2.0) * (a + 1.0) ** 2)


class _PowerLogNorm(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, c, s):
        z = _math.log(x) / s
        return c / (x * s) * _phi(z) * _norm_cdf(-z) ** (c - 1.0)

    def _cdf(self, x, c, s):
        return 1.0 - _norm_cdf(-_math.log(x) / s) ** c

    def _ppf(self, q, c, s):
        return _math.exp(-s * _norm_ppf((1.0 - q) ** (1.0 / c)))


class _PowerNorm(_LS):
    _shapes = 1

    def _pdf(self, x, c):
        return c * _phi(x) * _norm_cdf(-x) ** (c - 1.0)

    def _cdf(self, x, c):
        return 1.0 - _norm_cdf(-x) ** c

    def _ppf(self, q, c):
        return -_norm_ppf((1.0 - q) ** (1.0 / c))


class _RDist(_LS):
    _shapes = 1
    _support = (-1.0, 1.0)

    def _pdf(self, x, c):
        return _math.exp((c / 2.0 - 1.0) * _math.log1p(-x * x) - (_math.lgamma(0.5) + _math.lgamma(c / 2.0) - _math.lgamma(0.5 + c / 2.0)))

    def _cdf(self, x, c):
        g = _scalar(beta.cdf(x * x, 0.5, c / 2.0))
        return 0.5 + 0.5 * g if x >= 0 else 0.5 - 0.5 * g

    def _ppf(self, q, c):
        if q >= 0.5:
            return _math.sqrt(_scalar(beta.ppf(2.0 * q - 1.0, 0.5, c / 2.0)))
        return -_math.sqrt(_scalar(beta.ppf(1.0 - 2.0 * q, 0.5, c / 2.0)))


class _RecipInvGauss(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, mu):
        return 1.0 / _math.sqrt(2.0 * _math.pi * x) * _math.exp(-(1.0 - mu * x) ** 2 / (2.0 * x * mu * mu))

    def _cdf(self, x, mu):
        isqx = 1.0 / _math.sqrt(x)
        t1 = 1.0 - mu * x
        t2 = 1.0 + mu * x
        b = _norm_cdf(-isqx * t2 / mu)
        return _norm_cdf(-(isqx * t1 / mu)) - (_math.exp(2.0 / mu + _math.log(b)) if b > 0 else 0.0)


class _Semicircular(_LS):
    _support = (-1.0, 1.0)

    def _pdf(self, x):
        return 2.0 / _math.pi * _math.sqrt(1.0 - x * x)

    def _cdf(self, x):
        return 0.5 + (x * _math.sqrt(1.0 - x * x) + _math.asin(x)) / _math.pi


class _SkewCauchy(_LS):
    _shapes = 1

    def _pdf(self, x, a):
        s = 1.0 + a * (1.0 if x >= 0 else -1.0)
        return 1.0 / (_math.pi * ((x / s) ** 2 + 1.0))

    def _cdf(self, x, a):
        s = 1.0 + a * (1.0 if x >= 0 else -1.0)
        return (1.0 - a) / 2.0 + s / _math.pi * _math.atan(x / s)

    def _ppf(self, q, a):
        s = 1.0 - a if q < (1.0 - a) / 2.0 else 1.0 + a
        return s * _math.tan(_math.pi * (q - (1.0 - a) / 2.0) / s)


class _Trapezoid(_LS):
    _shapes = 2
    _support = (0.0, 1.0)

    def _pdf(self, x, c, d):
        h = 2.0 / (1.0 + d - c)
        if x < c:
            return h * x / c
        if x <= d:
            return h
        return h * (1.0 - x) / (1.0 - d)

    def _cdf(self, x, c, d):
        h = 2.0 / (1.0 + d - c)
        if x < c:
            return 0.5 * h * x * x / c
        if x <= d:
            return 0.5 * h * c + h * (x - c)
        return 1.0 - 0.5 * h * (1.0 - x) ** 2 / (1.0 - d)

    def _ppf(self, q, c, d):
        h = 2.0 / (1.0 + d - c)
        qc = 0.5 * h * c
        qd = qc + h * (d - c)
        if q < qc:
            return _math.sqrt(2.0 * q * c / h)
        if q <= qd:
            return c + (q - qc) / h
        return 1.0 - _math.sqrt(2.0 * (1.0 - q) * (1.0 - d) / h)


class _TruncExpon(_LS):
    _shapes = 1

    def _sup(self, b):
        return (0.0, b)

    def _pdf(self, x, b):
        return _math.exp(-x) / (-_math.expm1(-b))

    def _cdf(self, x, b):
        return -_math.expm1(-x) / (-_math.expm1(-b))

    def _ppf(self, q, b):
        return -_math.log1p(q * _math.expm1(-b))


class _TruncPareto(_LS):
    _shapes = 2

    def _sup(self, b, c):
        return (1.0, c)

    def _pdf(self, x, b, c):
        return b * x ** (-b - 1.0) / (1.0 - c ** (-b))

    def _cdf(self, x, b, c):
        return (1.0 - x ** (-b)) / (1.0 - c ** (-b))

    def _ppf(self, q, b, c):
        return (1.0 - q * (1.0 - c ** (-b))) ** (-1.0 / b)


class _TruncWeibullMin(_LS):
    _shapes = 3

    def _sup(self, c, a, b):
        return (a, b)

    def _pdf(self, x, c, a, b):
        den = _math.exp(-a ** c) - _math.exp(-b ** c)
        return c * x ** (c - 1.0) * _math.exp(-x ** c) / den

    def _cdf(self, x, c, a, b):
        return (_math.exp(-a ** c) - _math.exp(-x ** c)) / (_math.exp(-a ** c) - _math.exp(-b ** c))

    def _ppf(self, q, c, a, b):
        return (-_math.log(_math.exp(-a ** c) - q * (_math.exp(-a ** c) - _math.exp(-b ** c)))) ** (1.0 / c)


class _TukeyLambda(_LS):
    _shapes = 1

    def _sup(self, lam):
        if lam > 0:
            return (-1.0 / lam, 1.0 / lam)
        return (-_math.inf, _math.inf)

    @staticmethod
    def _Q(q, lam):
        if lam == 0:
            return _math.log(q / (1.0 - q))
        return (q ** lam - (1.0 - q) ** lam) / lam

    @staticmethod
    def _dQ(q, lam):
        if lam == 0:
            return 1.0 / (q * (1.0 - q))
        return q ** (lam - 1.0) + (1.0 - q) ** (lam - 1.0)

    def _ppf(self, q, lam):
        return self._Q(q, lam)

    def _cdf(self, x, lam):
        return _ppf_from_cdf(lambda u: self._Q(u, lam), x, 1e-300, 1.0 - 1e-16) if False else _bisect_unit(lambda u: self._Q(u, lam), x)

    def _pdf(self, x, lam):
        return 1.0 / self._dQ(self._cdf(x, lam), lam)

    def _isf(self, q, lam):
        return -self._ppf(q, lam)          # symmetric about 0

    def _stdmean(self, lam):
        return 0.0 if lam > -1.0 else _math.nan

    def _stdvar(self, lam):
        # 2/lam^2 (1/(1+2 lam) - Gamma(lam+1)^2/Gamma(2 lam+2)), lam > -1/2;
        # the logistic limit pi^2/3 at lam = 0
        if lam <= -0.5:
            return _math.inf
        if abs(lam) < 1e-8:
            return _math.pi ** 2 / 3.0
        return 2.0 / (lam * lam) * (1.0 / (1.0 + 2.0 * lam)
                                     - _math.exp(2.0 * _math.lgamma(lam + 1.0) - _math.lgamma(2.0 * lam + 2.0)))


def _bisect_unit(fn, target):
    """u in (0, 1) with fn(u) = target, fn increasing."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        try:
            v = fn(mid)
        except (ValueError, ZeroDivisionError, OverflowError):
            v = _math.inf if mid > 0.5 else -_math.inf
        if v < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class _WeibullMax(_LS):
    _shapes = 1
    _support = (-_math.inf, 0.0)

    def _pdf(self, x, c):
        return c * (-x) ** (c - 1.0) * _math.exp(-(-x) ** c)

    def _cdf(self, x, c):
        return _math.exp(-(-x) ** c)

    def _ppf(self, q, c):
        return -(-_math.log(q)) ** (1.0 / c)


class _WrapCauchy(_LS):
    _shapes = 1
    _support = (0.0, 2.0 * _math.pi)

    def _pdf(self, x, c):
        return (1.0 - c * c) / (2.0 * _math.pi * (1.0 + c * c - 2.0 * c * _math.cos(x)))

    def _cdf(self, x, c):
        # scipy's closed form, by half-angle tangent
        r = (1.0 + c) / (1.0 - c)
        if x < _math.pi:
            return _math.atan(r * _math.tan(0.5 * x)) / _math.pi
        return 1.0 - _math.atan(r * _math.tan(0.5 * (2.0 * _math.pi - x))) / _math.pi


alpha = _Alpha()
anglit = _Anglit()
arcsine = _Arcsine()
argus = _Argus()
betaprime = _BetaPrime()
bradford = _Bradford()
burr = _Burr()
burr12 = _Burr12()
chi = _Chi()
cosine = _Cosine()
crystalball = _CrystalBall()
dgamma = _DGamma()
dweibull = _DWeibull()
erlang = _Erlang()
exponnorm = _ExponNorm()
exponpow = _ExponPow()
exponweib = _ExponWeib()
fatiguelife = _FatigueLife()
fisk = _Fisk()
foldcauchy = _FoldCauchy()
foldnorm = _FoldNorm()
genexpon = _GenExpon()
gengamma = _GenGamma()
genhalflogistic = _GenHalfLogistic()
genlogistic = _GenLogistic()
gennorm = _GenNorm()
gibrat = _Gibrat()
gompertz = _Gompertz()
gumbel_r = _GumbelR()
gumbel_l = _GumbelL()
halfgennorm = _HalfGenNorm()
halflogistic = _HalfLogistic()
halfnorm = _HalfNorm()
hypsecant = _HypSecant()
invgauss = _InvGauss()
invweibull = _InvWeibull()
irwinhall = _IrwinHall()
jf_skew_t = _JFSkewT()
johnsonsb = _JohnsonSB()
johnsonsu = _JohnsonSU()
kappa3 = _Kappa3()
kappa4 = _Kappa4()
laplace_asymmetric = _LaplaceAsymmetric()
levy = _Levy()
levy_l = _LevyL()
loggamma = _LogGamma()
loglaplace = _LogLaplace()
lomax = _Lomax()
maxwell = _Maxwell()
mielke = _Mielke()
moyal = _Moyal()
nakagami = _Nakagami()
pearson3 = _Pearson3()
powerlaw = _PowerLaw()
powerlognorm = _PowerLogNorm()
powernorm = _PowerNorm()
rdist = _RDist()
recipinvgauss = _RecipInvGauss()
reciprocal = loguniform
semicircular = _Semicircular()
skewcauchy = _SkewCauchy()
trapezoid = _Trapezoid()
truncexpon = _TruncExpon()
truncpareto = _TruncPareto()
truncweibull_min = _TruncWeibullMin()
tukeylambda = _TukeyLambda()
vonmises_line = vonmises
weibull_max = _WeibullMax()
wrapcauchy = _WrapCauchy()


# ---------------------------------------------------- scipy parity: quadrature families

def _simpson_fixed(f, a, b, npan=2000):
    h = (b - a) / npan
    s = f(a) + f(b)
    for i in range(1, npan):
        s += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return s * h / 3.0


class _GaussHyper(_LS):
    _shapes = 4
    _support = (0.0, 1.0)

    @staticmethod
    def _kernel(x, a, b, c, z):
        return x ** (a - 1.0) * (1.0 - x) ** (b - 1.0) * (1.0 + z * x) ** (-c)

    def _norm(self, a, b, c, z):
        # B(a, b) 2F1(c, a; a + b; -z) as the integral of the kernel; the
        # substitution x = u^2 tames the endpoint when a < 1
        f = lambda u: 2.0 * u * self._kernel(u * u, a, b, c, z) if 0 < u < 1 else 0.0  # noqa: E731
        return _adaptive_simpson(f, 0.0, 1.0, 1e-12, 40)

    def _pdf(self, x, a, b, c, z):
        return self._kernel(x, a, b, c, z) / self._norm(a, b, c, z)

    def _cdf(self, x, a, b, c, z):
        f = lambda u: 2.0 * u * self._kernel(u * u, a, b, c, z) if 0 < u < 1 else 0.0  # noqa: E731
        return _adaptive_simpson(f, 0.0, _math.sqrt(x), 1e-12, 40) / self._norm(a, b, c, z)


class _GenHyperbolic(_LS):
    _shapes = 3

    def _pdf(self, x, p, a, b):
        g = _math.sqrt((a - b) * (a + b))
        s = _math.sqrt(1.0 + x * x)
        c = _math.exp(p * _math.log(g) - 0.5 * _math.log(2.0 * _math.pi) - (p - 0.5) * _math.log(a) - _math.log(_bessel_k(p, g)))
        return c * s ** (p - 0.5) * _bessel_k(p - 0.5, a * s) * _math.exp(b * x)


class _GenInvGauss(_LS):
    _shapes = 2
    _support = (0.0, _math.inf)

    def _pdf(self, x, p, b):
        return x ** (p - 1.0) * _math.exp(-0.5 * b * (x + 1.0 / x)) / (2.0 * _bessel_k(p, b))


class _NormInvGauss(_LS):
    _shapes = 2

    def _pdf(self, x, a, b):
        g = _math.sqrt((a + b) * (a - b))
        s = _math.hypot(1.0, x)
        return a / _math.pi * _bessel_k(1.0, a * s) * _math.exp(b * x + g) / s


class _Rice(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    def _pdf(self, x, b):
        return x * _math.exp(-0.5 * (x - b) ** 2) * _math.exp(-x * b) * _bessel_i(0.0, x * b)


    # X^2 ~ noncentral chi2(2, b^2)
    def _cdf(self, x, b):
        return _scalar(ncx2.cdf(x * x, 2, b * b)) if b > 0 else -_math.expm1(-0.5 * x * x)

    def _sf(self, x, b):
        return _scalar(ncx2.sf(x * x, 2, b * b)) if b > 0 else _math.exp(-0.5 * x * x)

    def _ppf(self, q, b):
        return _math.sqrt(_scalar(ncx2.ppf(q, 2, b * b))) if b > 0 else _math.sqrt(-2.0 * _math.log1p(-q))

    def _isf(self, q, b):
        return _math.sqrt(_scalar(ncx2.isf(q, 2, b * b))) if b > 0 else _math.sqrt(-2.0 * _math.log(q))

    def _stdmean(self, b):
        # sqrt(pi/2) L_{1/2}(-b^2/2)
        z = b * b / 4.0
        return _math.sqrt(_math.pi / 2.0) * _math.exp(-z) * (
            (1.0 + 2.0 * z) * _bessel_i(0, z) + 2.0 * z * _bessel_i(1, z))

    def _stdvar(self, b):
        m = self._stdmean(b)
        return 2.0 + b * b - m * m


class _RelBreitWigner(_LS):
    _shapes = 1
    _support = (0.0, _math.inf)

    @staticmethod
    def _k(rho):
        return 2.0 * _math.sqrt(2.0) * rho * rho * _math.sqrt(rho * rho + 1.0) / (
            _math.pi * _math.sqrt(rho * rho + rho * _math.sqrt(rho * rho + 1.0)))

    def _pdf(self, x, rho):
        return self._k(rho) / ((x * x - rho * rho) ** 2 + rho * rho)


class _Landau(_LS):
    """scipy's Landau: the stable law with alpha = 1, beta = 1 in the S1
    parameterisation (unit scale, zero location), so both the density and
    the distribution function come from the stable-law integrals."""

    def _pdf(self, x):
        return levy_stable._pdf0(x, 1.0, 1.0)

    def _cdf(self, x):
        return _bi.max(0.0, _bi.min(1.0, levy_stable._cdf0(x, 1.0, 1.0)))

class _LevyStable(_LS):
    """Stable law in scipy's default S1 parameterisation, by Nolan's (1997)
    integral representation in S0."""
    _shapes = 2

    @staticmethod
    def _theta0_zeta(alpha, beta):
        zeta = -beta * _math.tan(_math.pi * alpha / 2.0)
        theta0 = _math.atan(beta * _math.tan(_math.pi * alpha / 2.0)) / alpha
        return theta0, zeta

    @classmethod
    def _V(cls, theta, alpha, beta, theta0):
        c1 = _math.cos(alpha * theta0) ** (1.0 / (alpha - 1.0))
        return (c1 * (_math.cos(theta) / _math.sin(alpha * (theta0 + theta))) ** (alpha / (alpha - 1.0))
                * _math.cos(alpha * theta0 + (alpha - 1.0) * theta) / _math.cos(theta))

    def _pdf0(self, x0, alpha, beta):
        # density in S0 at x0
        if alpha == 1.0:
            if beta == 0.0:
                return 1.0 / (_math.pi * (1.0 + x0 * x0))
            if beta < 0:
                return self._pdf0(-x0, alpha, -beta)
            def f(theta):
                try:
                    v = 2.0 / _math.pi * ((_math.pi / 2.0 + beta * theta) / _math.cos(theta)) * _math.exp(
                        (_math.pi / 2.0 + beta * theta) * _math.tan(theta) / beta)
                    g = _math.exp(-_math.pi * x0 / (2.0 * beta)) * v
                    return g * _math.exp(-g) if g < 700 else 0.0
                except (OverflowError, ZeroDivisionError, ValueError):
                    return 0.0
            return _adaptive_simpson(f, -_math.pi / 2.0 + 1e-9, _math.pi / 2.0 - 1e-9, 1e-10, 30) / (2.0 * _bi.abs(beta))
        theta0, zeta = self._theta0_zeta(alpha, beta)
        if _bi.abs(x0 - zeta) < 1e-10:
            return _math.gamma(1.0 + 1.0 / alpha) * _math.cos(theta0) / (_math.pi * (1.0 + zeta * zeta) ** (0.5 / alpha))
        if x0 < zeta:
            return self._pdf0(-x0, alpha, -beta)
        d = x0 - zeta
        ex = alpha / (alpha - 1.0)

        def f(theta):
            try:
                v = self._V(theta, alpha, beta, theta0)
                g = d ** ex * v
                return g * _math.exp(-g) if g < 700 else 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0
        integral = _adaptive_simpson(f, -theta0 + 1e-9, _math.pi / 2.0 - 1e-9, 1e-10, 30)
        return alpha / (_math.pi * _bi.abs(alpha - 1.0) * d) * integral

    def _cdf0(self, x0, alpha, beta):
        if alpha == 1.0:
            if beta == 0.0:
                return 0.5 + _math.atan(x0) / _math.pi
            if beta < 0:
                return 1.0 - self._cdf0(-x0, alpha, -beta)
            def f(theta):
                try:
                    v = 2.0 / _math.pi * ((_math.pi / 2.0 + beta * theta) / _math.cos(theta)) * _math.exp(
                        (_math.pi / 2.0 + beta * theta) * _math.tan(theta) / beta)
                    g = _math.exp(-_math.pi * x0 / (2.0 * beta)) * v
                    return _math.exp(-g) if g < 700 else 0.0
                except (OverflowError, ZeroDivisionError, ValueError):
                    return 0.0
            return _adaptive_simpson(f, -_math.pi / 2.0 + 1e-9, _math.pi / 2.0 - 1e-9, 1e-10, 30) / _math.pi
        theta0, zeta = self._theta0_zeta(alpha, beta)
        if _bi.abs(x0 - zeta) < 1e-10:
            return 0.5 - theta0 / _math.pi
        if x0 < zeta:
            return 1.0 - self._cdf0(-x0, alpha, -beta)
        d = x0 - zeta
        ex = alpha / (alpha - 1.0)

        def f(theta):
            try:
                g = d ** ex * self._V(theta, alpha, beta, theta0)
                return _math.exp(-g) if g < 700 else 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0
        integral = _adaptive_simpson(f, -theta0 + 1e-9, _math.pi / 2.0 - 1e-9, 1e-10, 30)
        c1 = (0.5 - theta0 / _math.pi) if alpha < 1 else 1.0
        return c1 + (1.0 if alpha < 1 else -1.0) * integral / _math.pi

    def _to_s0(self, x, alpha, beta):
        if alpha == 1.0:
            return x
        return x - beta * _math.tan(_math.pi * alpha / 2.0)

    def _pdf(self, x, alpha, beta):
        return self._pdf0(self._to_s0(x, alpha, beta), alpha, beta)

    def _cdf(self, x, alpha, beta):
        return _bi.max(0.0, _bi.min(1.0, self._cdf0(self._to_s0(x, alpha, beta), alpha, beta)))


def _gauss_legendre(n):
    """Nodes and weights on [-1, 1] by Newton iteration on Legendre P_n."""
    cache = _GL_CACHE.get(n)
    if cache:
        return cache
    xs, ws = [], []
    for i in range(1, n + 1):
        x = _math.cos(_math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = p1 / dp
            x -= dx
            if _bi.abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2.0 / ((1.0 - x * x) * dp * dp))
    _GL_CACHE[n] = (xs, ws)
    return xs, ws


_GL_CACHE = {}


def _gl_integrate(f, a, b, n=64):
    xs, ws = _gauss_legendre(n)
    h = 0.5 * (b - a)
    m = 0.5 * (a + b)
    return h * _math.fsum(w * f(m + h * x) for x, w in zip(xs, ws))


class _StudentizedRange(_LS):
    """Studentized range: the double integral of Lund & Lund / Harter,
    inner over the normal location, outer over the scale factor
    s = sqrt(chi2_df / df), both by Gauss-Legendre quadrature."""
    _shapes = 2
    _support = (0.0, _math.inf)

    def _eff_range(self, *sh):
        return (1e-6, 40.0)

    @staticmethod
    def _s_density(s, df):
        return _math.exp(0.5 * df * _math.log(df) - _math.lgamma(0.5 * df) - (0.5 * df - 1.0) * _math.log(2.0)
                         + (df - 1.0) * _math.log(s) - 0.5 * df * s * s)

    def _inner_cdf(self, q, s, k):
        qs = q * s
        if qs > 40.0:
            return 1.0
        # the integrand lives on z in [-8 - qs, 8]; split at the middle so
        # the two humps each get their own nodes
        f = lambda z: _phi(z) * (_norm_cdf(z + qs) - _norm_cdf(z)) ** (k - 1.0)  # noqa: E731
        lo, hi = -8.0 - qs, 8.0
        mid = 0.5 * (lo + hi)
        return k * (_gl_integrate(f, lo, mid, 64) + _gl_integrate(f, mid, hi, 64))

    def _inner_pdf(self, q, s, k):
        qs = q * s
        if qs > 40.0:
            return 0.0
        f = lambda z: _phi(z) * _phi(z + qs) * (_norm_cdf(z + qs) - _norm_cdf(z)) ** (k - 2.0)  # noqa: E731
        lo, hi = -8.0 - qs, 8.0
        mid = 0.5 * (lo + hi)
        return k * (k - 1.0) * s * (_gl_integrate(f, lo, mid, 64) + _gl_integrate(f, mid, hi, 64))

    def _outer(self, q, k, df, inner):
        if df > 1e5:
            return inner(q, 1.0, k)
        # s has mean ~1 and sd ~ 1/sqrt(2 df): integrate over +-12 sd
        sd = 1.0 / _math.sqrt(2.0 * df)
        lo, hi = _bi.max(1e-6, 1.0 - 12.0 * sd), 1.0 + 12.0 * sd
        g = lambda s: self._s_density(s, df) * inner(q, s, k)  # noqa: E731
        pieces = 8
        step = (hi - lo) / pieces
        return _math.fsum(_gl_integrate(g, lo + i * step, lo + (i + 1) * step, 32) for i in range(pieces))

    def _cdf(self, q, k, df):
        return self._outer(q, k, df, self._inner_cdf)

    def _pdf(self, q, k, df):
        return self._outer(q, k, df, self._inner_pdf)

class _DParetoLogNorm(_LS):
    """Reed's double Pareto-lognormal with scipy's (u, s, a, b) shapes."""
    _shapes = 4
    _support = (0.0, _math.inf)

    @staticmethod
    def _R(t):
        # Mills ratio Phi_c(t) / phi(t), stable in both tails
        if t > 30.0:
            return (1.0 / t) * (1.0 - 1.0 / (t * t) + 3.0 / t ** 4)
        return _norm_cdf(-t) / _phi(t)

    def _pdf(self, y, u, s, a, b):
        z = (_math.log(y) - u) / s
        return a * b / (a + b) / y * _phi(z) * (self._R(a * s - z) + self._R(b * s + z))

    def _cdf(self, y, u, s, a, b):
        z = (_math.log(y) - u) / s
        return _norm_cdf(z) - _phi(z) * (b * self._R(a * s - z) - a * self._R(b * s + z)) / (a + b)


gausshyper = _GaussHyper()
genhyperbolic = _GenHyperbolic()
geninvgauss = _GenInvGauss()
norminvgauss = _NormInvGauss()
rice = _Rice()
rel_breitwigner = _RelBreitWigner()
landau = _Landau()
levy_stable = _LevyStable()
studentized_range = _StudentizedRange()
dpareto_lognorm = _DParetoLogNorm()


# ---------------------------------------------------- scipy parity: discrete families

class _Disc(_Dist):
    """A discrete family: subclasses give ``_pmf(k, *shape)`` and the
    support ``_sup(*shape)``; cdf sums the mass, ppf walks it."""
    _discrete = True

    def _sup(self, *sh):
        return self._support

    def _bounds(self, *args, **kw):
        return tuple(float(v) for v in self._sup(*args))

    def pmf(self, k, *args):
        lo, hi = self._sup(*args)

        def one(v):
            if v != v:
                return _math.nan
            if v < lo or v > hi or float(v) != _math.floor(v):
                return 0.0
            return self._pmf(int(v), *args)
        return _maybe_map(one, k)

    def cdf(self, k, *args):
        lo, hi = self._sup(*args)

        def one(v):
            if v != v:
                return _math.nan
            if v < lo:
                return 0.0
            if v >= hi:
                return 1.0
            kk = int(_math.floor(v))
            start = int(lo) if lo > -_math.inf else self._lower_start(*args)
            return _bi.min(1.0, _math.fsum(self._pmf(i, *args) for i in range(start, kk + 1)))
        return _maybe_map(one, k)

    def ppf(self, q, *args):
        lo, hi = self._sup(*args)
        start = int(lo) if lo > -_math.inf else self._lower_start(*args)

        def one(p):
            if p != p or p < 0.0 or p > 1.0:
                return _math.nan
            if p == 0.0:
                return float(start - 1)
            return _discrete_ppf(p, lambda i: self._pmf(i, *args), start, None if hi == _math.inf else int(hi))
        return _maybe_map(one, q)
    pmf._edge_wrapped = True
    cdf._edge_wrapped = True
    ppf._edge_wrapped = True


class _BetaNBinom(_Disc):
    _support = (0.0, _math.inf)

    def _pmf(self, k, n, a, b):
        return _math.exp(_log_comb(n + k - 1, k) + _math.lgamma(a + n) + _math.lgamma(b + k) - _math.lgamma(a + b + n + k)
                         + _math.lgamma(a + b) - _math.lgamma(a) - _math.lgamma(b))


class _Boltzmann(_Disc):
    def _sup(self, lam, N):
        return (0.0, float(N) - 1.0)

    def _pmf(self, k, lam, N):
        return (1.0 - _math.exp(-lam)) * _math.exp(-lam * k) / (1.0 - _math.exp(-lam * N))


class _DLaplace(_Disc):
    _support = (-_math.inf, _math.inf)

    def _lower_start(self, a):
        return -int(60.0 / a) - 5

    def _pmf(self, k, a):
        return _math.tanh(0.5 * a) * _math.exp(-a * _bi.abs(k))


class _LogSer(_Disc):
    _support = (1.0, _math.inf)

    def _pmf(self, k, p):
        return -p ** k / (k * _math.log1p(-p))


class _NCHypergeomFisher(_Disc):
    def _sup(self, M, n, N, odds):
        return (float(_bi.max(0, N - (M - n))), float(_bi.min(n, N)))

    def _weights(self, M, n, N, odds):
        lo, hi = self._sup(M, n, N, odds)
        ks = range(int(lo), int(hi) + 1)
        w = [_math.exp(_log_comb(int(n), k) + _log_comb(int(M - n), int(N) - k) + k * _math.log(odds)) for k in ks]
        tot = _math.fsum(w)
        return {k: v / tot for k, v in zip(ks, w)}

    def _pmf(self, k, M, n, N, odds):
        return self._weights(M, n, N, odds).get(k, 0.0)


class _NCHypergeomWallenius(_Disc):
    def _sup(self, M, n, N, odds):
        return (float(_bi.max(0, N - (M - n))), float(_bi.min(n, N)))

    def _raw(self, k, M, n, N, odds):
        D = odds * (n - k) + (M - n - (N - k))
        if D <= 0:
            return 1.0 if k == N else 0.0

        def f(t):
            if t <= 0.0 or t >= 1.0:
                return 0.0
            return (1.0 - t ** (odds / D)) ** k * (1.0 - t ** (1.0 / D)) ** (N - k)
        integral = _adaptive_simpson(f, 0.0, 1.0, 1e-14, 45)
        return _math.exp(_log_comb(n, k) + _log_comb(M - n, N - k)) * integral

    def _pmf(self, k, M, n, N, odds):
        M, n, N = int(M), int(n), int(N)
        cache = self.__dict__.setdefault("_wcache", {})
        key = (M, n, N, float(odds))
        if key not in cache:
            lo, hi = self._sup(M, n, N, odds)
            raw = {i: self._raw(i, M, n, N, odds) for i in range(int(lo), int(hi) + 1)}
            tot = _math.fsum(raw.values())
            cache[key] = {i: v / tot for i, v in raw.items()}
        return cache[key].get(k, 0.0)


class _NHypergeom(_Disc):
    def _sup(self, M, n, r):
        return (0.0, float(n))

    def _pmf(self, k, M, n, r):
        M, n, r = int(M), int(n), int(r)
        if M - r - k < n - k:
            return 0.0
        return _math.exp(_log_comb(k + r - 1, k) + _log_comb(M - r - k, n - k) - _log_comb(M, n))


class _Planck(_Disc):
    _support = (0.0, _math.inf)

    def _pmf(self, k, lam):
        return (1.0 - _math.exp(-lam)) * _math.exp(-lam * k)

    def cdf(self, k, lam):
        return _maybe_map(lambda v: 0.0 if v < 0 else 1.0 - _math.exp(-lam * (_math.floor(v) + 1.0)), k)

    def ppf(self, q, lam):
        return _maybe_map(lambda p: float(_bi.max(0, _math.ceil(-_math.log1p(-p) / lam - 1.0))), q)


class _PoissonBinom(_Disc):
    def _sup(self, p):
        return (0.0, float(len(list(p))))

    def _table(self, p):
        probs = [float(v) for v in p]
        dp = [1.0]
        for pi in probs:
            nxt = [0.0] * (len(dp) + 1)
            for i, v in enumerate(dp):
                nxt[i] += v * (1.0 - pi)
                nxt[i + 1] += v * pi
            dp = nxt
        return dp

    def _pmf(self, k, p):
        t = self._table(p)
        return t[k] if 0 <= k < len(t) else 0.0


class _YuleSimon(_Disc):
    _support = (1.0, _math.inf)

    def _pmf(self, k, alpha):
        return alpha * _math.exp(_math.lgamma(k) + _math.lgamma(alpha + 1.0) - _math.lgamma(k + alpha + 1.0))

    def cdf(self, k, alpha):
        return _maybe_map(lambda v: 0.0 if v < 1 else 1.0 - _math.floor(v) * _math.exp(
            _math.lgamma(_math.floor(v)) + _math.lgamma(alpha + 1.0) - _math.lgamma(_math.floor(v) + alpha + 1.0)), k)


class _Zipfian(_Disc):
    def _sup(self, a, n):
        return (1.0, float(n))

    def _pmf(self, k, a, n):
        h = _math.fsum(i ** (-a) for i in range(1, int(n) + 1))
        return k ** (-a) / h


betanbinom = _BetaNBinom()
boltzmann = _Boltzmann()
dlaplace = _DLaplace()
logser = _LogSer()
nchypergeom_fisher = _NCHypergeomFisher()
nchypergeom_wallenius = _NCHypergeomWallenius()
nhypergeom = _NHypergeom()
planck = _Planck()
poisson_binom = _PoissonBinom()
yulesimon = _YuleSimon()
zipfian = _Zipfian()
