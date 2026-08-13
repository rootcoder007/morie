# morie.fn -- shared core (rootcoder007/morie)
"""R-compatible random number generation.

Reproduces R's default stream exactly: the Mersenne-Twister generator
seeded the way ``set.seed()`` seeds it, ``runif()``, and the
``R_unif_index`` rejection sampler that ``sample()`` has used since
R 3.6.0.  With this, any bootstrap we port from WRS produces the same
numbers R does, so those functions can be anchored digit for digit
instead of only structurally.

Source: R sources, ``src/main/RNG.c`` (RNG_Init, MT_sgenrand,
MT_genrand, fixup, unif_rand) and ``src/main/random.c`` /
``src/nmath/...`` for R_unif_index and rbits.  The Mersenne Twister
itself is Matsumoto, M. and Nishimura, T. (1998), "Mersenne twister:
a 623-dimensionally equidistributed uniform pseudo-random number
generator", ACM Transactions on Modeling and Computer Simulation
8(1), 3-30, doi:10.1145/272991.272995. PDF not in hand (the ACM
Digital Library serves HTML); cited from bibliographic details.

No external dependencies -- plain integer arithmetic.
"""

import math

__all__ = ["RRandom", "set_seed", "runif", "sample_int", "sample"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.3283064365386963e-10        # 1 / 2^32, R's i2_32m1


class RRandom:
    """R's Mersenne-Twister stream.

    ``RRandom(seed)`` corresponds to ``set.seed(seed)`` in R with the
    default ``kind = "Mersenne-Twister"`` and
    ``sample.kind = "Rejection"``.
    """

    def __init__(self, seed=None):
        self.mt = [0] * _N
        self.mti = _N + 1
        if seed is not None:
            self.set_seed(seed)

    # ---------------------------------------------------------------
    def set_seed(self, seed):
        """R's ``set.seed``: scramble, then fill the state.

        RNG_Init applies the LCG ``seed = 69069 * seed + 1`` fifty times
        as "initial scrambling", then fills all 625 state words with
        further iterates.  i_seed[0] holds mti and is overwritten by
        FixupSeeds with 624; i_seed[1..624] become the MT state.
        """
        s = int(seed) & _MASK32
        for _ in range(50):                       # initial scrambling
            s = (69069 * s + 1) & _MASK32
        # i_seed[0] is mti in R, but FixupSeeds resets it to 624
        s = (69069 * s + 1) & _MASK32
        for j in range(_N):
            s = (69069 * s + 1) & _MASK32
            self.mt[j] = s
        self.mti = _N                             # FixupSeeds(initial=1)

    # ---------------------------------------------------------------
    def _genrand_int32(self):
        """MT_genrand: the standard MT19937 tempered output."""
        mt, mag01 = self.mt, (0, _MATRIX_A)
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & _MASK32

    # ---------------------------------------------------------------
    def unif_rand(self):
        """R's ``unif_rand`` for Mersenne-Twister, including ``fixup``.

        fixup() nudges the value strictly inside (0, 1); R's constants
        are 2.328306437080797e-10 and 1 - that.
        """
        v = self._genrand_int32() * _I2_32M1
        # fixup: ensure 0 < v < 1
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    runif_one = unif_rand

    def runif(self, n=1, lo=0.0, hi=1.0):
        """R's ``runif(n, min, max)``."""
        out = [lo + (hi - lo) * self.unif_rand() for _ in range(int(n))]
        return out

    # ---------------------------------------------------------------
    def _rbits(self, bits):
        """R's ``rbits``: build an integer 16 bits at a time.

        Note the loop condition is ``n <= bits``, so it draws one more
        16-bit chunk than strictly needed; reproducing that is essential
        because it consumes the same number of uniforms R does.
        """
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn):
        """R's ``R_unif_index`` with sample.kind = "Rejection".

        Draws from the integers below the next power of two and rejects
        anything too large, which is what makes ``sample()`` unbiased.
        """
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        dv = self._rbits(bits)
        while dn <= dv:
            dv = self._rbits(bits)
        return dv

    # ---------------------------------------------------------------
    def sample_int(self, n, size=None, replace=False):
        """R's ``sample.int(n, size, replace)``, 1-based like R."""
        n = int(n)
        k = n if size is None else int(size)
        if replace:
            return [int(self.unif_index(n)) + 1 for _ in range(k)]
        if k > n:
            raise ValueError("cannot take a sample larger than the "
                             "population when replace = FALSE")
        pool = list(range(1, n + 1))
        m = n
        out = []
        for _ in range(k):
            j = int(self.unif_index(m))
            out.append(pool[j])
            pool[j] = pool[m - 1]
            m -= 1
        return out

    def sample(self, x, size=None, replace=False):
        """R's ``sample(x, size, replace)`` for a vector ``x``."""
        seq = list(x)
        idx = self.sample_int(len(seq), size, replace)
        return [seq[i - 1] for i in idx]


_GLOBAL = RRandom()


def set_seed(seed):
    """Seed the module-level stream, as ``set.seed`` does in R."""
    _GLOBAL.set_seed(seed)


def runif(n=1, lo=0.0, hi=1.0):
    """Draw from the module-level stream."""
    return _GLOBAL.runif(n, lo, hi)


def sample_int(n, size=None, replace=False):
    """``sample.int`` on the module-level stream."""
    return _GLOBAL.sample_int(n, size, replace)


def sample(x, size=None, replace=False):
    """``sample`` on the module-level stream."""
    return _GLOBAL.sample(x, size, replace)


# ---------------------------------------------------------------- d/p/q/r
# R-named distribution functions.  Kept here so callers stop hand-rolling
# Box-Muller, and pointed at AS 241 rather than the coarser Acklam
# approximation that also lives in this package.

import math as _math


def dnorm(x, mean=0.0, sd=1.0, log=False):
    """Normal density.  Scalar or sequence, following the input."""
    if sd <= 0:
        raise ValueError("sd must be positive")

    def one(v):
        z = (float(v) - mean) / sd
        lg = -0.5 * z * z - _math.log(sd) - 0.5 * _math.log(2.0 * _math.pi)
        return lg if log else _math.exp(lg)

    if isinstance(x, (list, tuple)):
        return [one(v) for v in x]
    return one(x)


def pnorm(q, mean=0.0, sd=1.0, lower_tail=True, log=False):
    """Normal cdf via erfc, which keeps the far tail accurate.

    Computing the upper tail as 1 - Phi(q) loses every significant digit
    once Phi(q) rounds to 1, so the upper tail is taken from erfc directly.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")

    def one(v):
        z = (float(v) - mean) / sd
        p = (0.5 * _math.erfc(-z / _math.sqrt(2.0)) if lower_tail
             else 0.5 * _math.erfc(z / _math.sqrt(2.0)))
        if log:
            return -745.0 if p <= 0.0 else _math.log(p)
        return p

    if isinstance(q, (list, tuple)):
        return [one(v) for v in q]
    return one(q)


def qnorm(p, mean=0.0, sd=1.0, lower_tail=True):
    """Normal quantile: Wichura's AS 241 (PPND16), about 1e-16."""
    from ._rng import normal_quantile as _ppnd16
    if sd <= 0:
        raise ValueError("sd must be positive")

    def one(v):
        u = float(v) if lower_tail else 1.0 - float(v)
        if not (0.0 < u < 1.0):
            raise ValueError("p must lie strictly inside (0, 1)")
        z = _ppnd16(u)
        z = float(z if not hasattr(z, "_flat") else list(z._flat())[0])
        return mean + sd * z

    if isinstance(p, (list, tuple)):
        return [one(v) for v in p]
    return one(p)


def rnorm(n, mean=0.0, sd=1.0):
    """Normal draws by inversion of the uniform stream.

    Inversion rather than Box-Muller so that draw k depends only on
    uniform k: the stream is stable under a change of n, and one draw is
    not paired with the next.
    """
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [qnorm(min(max(u, 1e-300), 1.0 - 1e-16), mean, sd) for u in us]


def rexp(n, rate=1.0):
    """Exponential draws by inversion: -log(1 - u)/rate."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [-_math.log1p(-min(u, 1.0 - 1e-16)) / rate for u in us]


def dexp(x, rate=1.0, log=False):
    """Exponential density."""
    if rate <= 0:
        raise ValueError("rate must be positive")

    def one(v):
        v = float(v)
        if v < 0:
            return -_math.inf if log else 0.0
        lg = _math.log(rate) - rate * v
        return lg if log else _math.exp(lg)

    if isinstance(x, (list, tuple)):
        return [one(v) for v in x]
    return one(x)


def pexp(q, rate=1.0, lower_tail=True):
    """Exponential cdf; the upper tail is exp(-rate q), not 1 - cdf."""
    if rate <= 0:
        raise ValueError("rate must be positive")

    def one(v):
        v = float(v)
        if v < 0:
            return 0.0 if lower_tail else 1.0
        return (-_math.expm1(-rate * v) if lower_tail
                else _math.exp(-rate * v))

    if isinstance(q, (list, tuple)):
        return [one(v) for v in q]
    return one(q)


def qexp(p, rate=1.0):
    """Exponential quantile."""
    if rate <= 0:
        raise ValueError("rate must be positive")

    def one(v):
        v = float(v)
        if not (0.0 <= v < 1.0):
            raise ValueError("p must lie in [0, 1)")
        return -_math.log1p(-v) / rate

    if isinstance(p, (list, tuple)):
        return [one(v) for v in p]
    return one(p)


# ------------------------------------------------- the rest of d/p/q/r
# Built on _stats_core's incomplete gamma and incomplete beta so that the
# special functions have exactly one implementation in the package.


def _sc():
    from . import _stats_core as s
    return s


def _bisect_q(cdf, p, lo, hi, tol=1e-12):
    """Smallest x with cdf(x) >= p, by bisection on a monotone cdf."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return lo
        if p == 1.0:
            return hi
        raise ValueError("p must lie in [0, 1]")
    while cdf(hi) < p:
        hi = hi * 2.0 + 1.0
        if hi > 1e300:
            raise ValueError("cdf never reaches p")
    while cdf(lo) > p:
        lo = lo * 2.0 - 1.0
        if lo < -1e300:
            raise ValueError("cdf never falls below p")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= tol * max(1.0, abs(hi)):
            break
    x = 0.5 * (lo + hi)
    # Bisection converges on the interval, not on the value.  Where the cdf
    # is flat -- around the median of a symmetric law -- the bracket can
    # close while x is still ~1e-8 from the root, so qt(0.5, df) came back
    # as -2.1e-08 instead of 0.  A couple of Newton steps on the residual
    # land it, and snapping a near-zero result to exact zero keeps the
    # symmetry qt(0.5) == 0 that callers reasonably assume.
    for _ in range(3):
        fx = cdf(x) - p
        h = 1e-6 * max(1.0, abs(x))
        d = (cdf(x + h) - cdf(x - h)) / (2.0 * h)
        if d <= 0.0 or not _math.isfinite(d):
            break
        step = fx / d
        if not _math.isfinite(step) or abs(step) > abs(hi - lo) + 1.0:
            break
        x -= step
    if abs(x) < 1e-11:
        x = 0.0
    return x


def _elem(f, x):
    return [f(v) for v in x] if isinstance(x, (list, tuple)) else f(x)


# ---- gamma ----------------------------------------------------------

def dgamma(x, shape, rate=1.0, log=False):
    if shape <= 0 or rate <= 0:
        raise ValueError("shape and rate must be positive")

    def one(v):
        v = float(v)
        if v < 0:
            return -_math.inf if log else 0.0
        if v == 0:
            return (0.0 if shape > 1 else _math.inf) if not log else -_math.inf
        lg = (shape * _math.log(rate) + (shape - 1) * _math.log(v)
              - rate * v - _math.lgamma(shape))
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def pgamma(q, shape, rate=1.0, lower_tail=True):
    s = _sc()

    def one(v):
        v = float(v)
        if v <= 0:
            return 0.0 if lower_tail else 1.0
        p = s._gammainc_p(shape, rate * v)
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qgamma(p, shape, rate=1.0):
    def cdf(v):
        return pgamma(v, shape, rate)
    return _elem(lambda pp: _bisect_q(cdf, float(pp), 0.0, 1.0), p)


# ---- chi-square (gamma with shape k/2, rate 1/2) ---------------------

def dchisq(x, df, log=False):
    return dgamma(x, df / 2.0, 0.5, log=log)


def pchisq(q, df, lower_tail=True):
    return pgamma(q, df / 2.0, 0.5, lower_tail=lower_tail)


def qchisq(p, df):
    return qgamma(p, df / 2.0, 0.5)


# ---- Poisson --------------------------------------------------------

def dpois(x, lam, log=False):
    if lam < 0:
        raise ValueError("lambda must be non-negative")

    def one(v):
        k = int(round(float(v)))
        if k < 0:
            return -_math.inf if log else 0.0
        lg = k * _math.log(lam) - lam - _math.lgamma(k + 1) if lam > 0 else (
            0.0 if k == 0 else -_math.inf)
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def ppois(q, lam, lower_tail=True):
    """P(X <= k) = Q(k+1, lambda), the UPPER regularised incomplete gamma."""
    s = _sc()

    def one(v):
        k = _math.floor(float(v))
        if k < 0:
            return 0.0 if lower_tail else 1.0
        p = 1.0 - s._gammainc_p(k + 1.0, lam)
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qpois(p, lam):
    """Smallest k with P(X <= k) >= p, as R defines it."""
    def one(pp):
        pp = float(pp)
        if not (0.0 <= pp <= 1.0):
            raise ValueError("p must lie in [0, 1]")
        k = 0
        while ppois(k, lam) < pp - 1e-15:
            k += 1
            if k > 10_000_000:
                raise ValueError("qpois failed to converge")
        return k
    return _elem(one, p)


# ---- binomial -------------------------------------------------------

def dbinom(x, size, prob, log=False):
    if not (0.0 <= prob <= 1.0):
        raise ValueError("prob must lie in [0, 1]")

    def one(v):
        k = int(round(float(v)))
        if k < 0 or k > size:
            return -_math.inf if log else 0.0
        lg = (_math.lgamma(size + 1) - _math.lgamma(k + 1)
              - _math.lgamma(size - k + 1))
        if prob == 0.0:
            lg = 0.0 if k == 0 else -_math.inf
        elif prob == 1.0:
            lg = 0.0 if k == size else -_math.inf
        else:
            lg += k * _math.log(prob) + (size - k) * _math.log1p(-prob)
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def pbinom(q, size, prob, lower_tail=True):
    """P(X <= k) = I_{1-p}(n-k, k+1) -- the regularised incomplete beta."""
    s = _sc()

    def one(v):
        k = _math.floor(float(v))
        if k < 0:
            return 0.0 if lower_tail else 1.0
        if k >= size:
            return 1.0 if lower_tail else 0.0
        p = s._betainc(size - k, k + 1.0, 1.0 - prob)
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qbinom(p, size, prob):
    def one(pp):
        pp = float(pp)
        if not (0.0 <= pp <= 1.0):
            raise ValueError("p must lie in [0, 1]")
        k = 0
        while k < size and pbinom(k, size, prob) < pp - 1e-15:
            k += 1
        return k
    return _elem(one, p)


# ---- beta -----------------------------------------------------------

def dbeta(x, shape1, shape2, log=False):
    if shape1 <= 0 or shape2 <= 0:
        raise ValueError("shape parameters must be positive")

    def one(v):
        v = float(v)
        if v < 0 or v > 1:
            return -_math.inf if log else 0.0
        if v in (0.0, 1.0):
            return 0.0 if not log else -_math.inf
        lg = ((shape1 - 1) * _math.log(v) + (shape2 - 1) * _math.log1p(-v)
              + _math.lgamma(shape1 + shape2) - _math.lgamma(shape1)
              - _math.lgamma(shape2))
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def pbeta(q, shape1, shape2, lower_tail=True):
    s = _sc()

    def one(v):
        v = float(v)
        if v <= 0:
            return 0.0 if lower_tail else 1.0
        if v >= 1:
            return 1.0 if lower_tail else 0.0
        p = s._betainc(shape1, shape2, v)
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qbeta(p, shape1, shape2):
    def cdf(v):
        return pbeta(min(max(v, 0.0), 1.0), shape1, shape2)
    return _elem(lambda pp: _bisect_q(cdf, float(pp), 0.0, 1.0), p)


# ---- Student t ------------------------------------------------------

def dt(x, df, log=False):
    if df <= 0:
        raise ValueError("df must be positive")

    def one(v):
        v = float(v)
        lg = (_math.lgamma((df + 1) / 2.0) - _math.lgamma(df / 2.0)
              - 0.5 * _math.log(df * _math.pi)
              - (df + 1) / 2.0 * _math.log1p(v * v / df))
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def pt(q, df, lower_tail=True):
    s = _sc()

    def one(v):
        v = float(v)
        xb = df / (df + v * v)
        half = 0.5 * s._betainc(df / 2.0, 0.5, xb)
        p = half if v <= 0 else 1.0 - half
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qt(p, df):
    # symmetric law: qt(p) = -qt(1 - p), qt(0.5) = 0 exactly.  betainc's
    # 1 - xb underflows for |v| < ~1e-8, so the cdf plateaus at 0.5 there
    # and root-finding is blind inside the plateau; symmetry removes it.
    def cdf(v):
        return pt(v, df)

    def one(pp):
        pp = float(pp)
        if pp == 0.5:
            return 0.0
        if pp < 0.5:
            return -_bisect_q(cdf, 1.0 - pp, -1.0, 1.0)
        return _bisect_q(cdf, pp, -1.0, 1.0)
    return _elem(one, p)


# ---- F --------------------------------------------------------------

def df_(x, df1, df2, log=False):
    """F density.  Named df_ because df is universally a parameter name."""
    def one(v):
        v = float(v)
        if v <= 0:
            return -_math.inf if log else 0.0
        lg = (0.5 * df1 * _math.log(df1 * v) + 0.5 * df2 * _math.log(df2)
              - 0.5 * (df1 + df2) * _math.log(df1 * v + df2)
              - _math.log(v) - _math.lgamma(df1 / 2.0)
              - _math.lgamma(df2 / 2.0) + _math.lgamma((df1 + df2) / 2.0))
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def pf(q, df1, df2, lower_tail=True):
    s = _sc()

    def one(v):
        v = float(v)
        if v <= 0:
            return 0.0 if lower_tail else 1.0
        xb = df1 * v / (df1 * v + df2)
        p = s._betainc(df1 / 2.0, df2 / 2.0, xb)
        return p if lower_tail else 1.0 - p
    return _elem(one, q)


def qf(p, df1, df2):
    def cdf(v):
        return pf(v, df1, df2)
    return _elem(lambda pp: _bisect_q(cdf, float(pp), 0.0, 1.0), p)


# ---- log-normal -----------------------------------------------------

def dlnorm(x, meanlog=0.0, sdlog=1.0, log=False):
    def one(v):
        v = float(v)
        if v <= 0:
            return -_math.inf if log else 0.0
        lg = (dnorm(_math.log(v), meanlog, sdlog, log=True) - _math.log(v))
        return lg if log else _math.exp(lg)
    return _elem(one, x)


def plnorm(q, meanlog=0.0, sdlog=1.0, lower_tail=True):
    def one(v):
        v = float(v)
        if v <= 0:
            return 0.0 if lower_tail else 1.0
        return pnorm(_math.log(v), meanlog, sdlog, lower_tail=lower_tail)
    return _elem(one, q)


def qlnorm(p, meanlog=0.0, sdlog=1.0):
    return _elem(lambda pp: _math.exp(qnorm(float(pp), meanlog, sdlog)), p)


# ---- draws by inversion ---------------------------------------------

def rbinom(n, size, prob):
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [qbinom(u, size, prob) for u in us]


def rpois(n, lam):
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [qpois(u, lam) for u in us]


def rgamma(n, shape, rate=1.0):
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [qgamma(min(max(u, 1e-12), 1 - 1e-12), shape, rate) for u in us]


def rbeta(n, shape1, shape2):
    us = runif(int(n))
    us = list(us._flat()) if hasattr(us, "_flat") else list(us)
    return [qbeta(min(max(u, 1e-12), 1 - 1e-12), shape1, shape2) for u in us]


def rchisq(n, df):
    return rgamma(n, df / 2.0, 0.5)


def rlnorm(n, meanlog=0.0, sdlog=1.0):
    return [_math.exp(v) for v in rnorm(n, meanlog, sdlog)]
