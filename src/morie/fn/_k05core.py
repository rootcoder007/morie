"""Slice-local helpers for the k05 batch.

Holds one thing the rest of the batch needs and cannot get elsewhere: a
working counter-based RNG.

``morie.fn._rng`` implements the package's Philox4x32-10 / AS 241
generator against a NumPy-shaped backend, and the de-numpy'd
``_array_core`` no longer carries ``uint64``, so every entry point in
that module raises ``AttributeError`` today. The R arm's copy
(``aaa_rng_native.R``) is written in plain arithmetic and still works.
So the algorithm below is a transcription of the *R* arm -- same
constants, same counter layout, same open-interval mapping -- done in
plain Python integers, which have no width limit and so need no
``uint64``. That keeps the two arms bit-identical while touching no
shared core.

If ``_rng`` is repaired, delete this and import it instead.
"""

from math import log, sqrt

__all__ = []

_MASK32 = 0xFFFFFFFF
_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9          # golden ratio
_W1 = 0xBB67AE85          # sqrt(3) - 1
_ROUNDS = 10


def philox4x32(ctr, key, rounds=_ROUNDS):
    """The Philox4x32 bijection on one counter block. Four words in, four out."""
    c0, c1, c2, c3 = (int(v) & _MASK32 for v in ctr)
    k0 = int(key[0]) & _MASK32
    k1 = int(key[1]) & _MASK32
    for r in range(rounds):
        p0 = _M0 * c0
        p1 = _M1 * c2
        hi0, lo0 = (p0 >> 32) & _MASK32, p0 & _MASK32
        hi1, lo1 = (p1 >> 32) & _MASK32, p1 & _MASK32
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK32, lo1, (hi0 ^ c3 ^ k1) & _MASK32, lo0
        if r + 1 < rounds:
            k0 = (k0 + _W0) & _MASK32
            k1 = (k1 + _W1) & _MASK32
    return c0, c1, c2, c3


def runif(n, seed=0, stream=0):
    """``n`` uniforms in the OPEN interval (0, 1), as a plain list.

    Open interval on purpose: a normal quantile at 0 or 1 is infinite,
    and ``(w + 0.5) / 2**32`` can reach neither endpoint.
    """
    n = int(n)
    if n < 0:
        raise ValueError("`n` must be non-negative")
    out = []
    seed = int(seed)
    key = (seed & _MASK32, (seed >> 32) & _MASK32)
    blocks = (n + 3) // 4
    for j in range(blocks):
        w = philox4x32((j & _MASK32, (j >> 32) & _MASK32, int(stream) & _MASK32, 0), key)
        out.extend((v + 0.5) / 4294967296.0 for v in w)
    return out[:n]


# --- Wichura (1988) AS 241, PPND16 ----------------------------------------
_A = (3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
      1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3)
_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2, 5.3941960214247511077e3,
      2.1213794301586595867e4, 3.9307895800092710610e4, 2.8729085735721942674e4,
      5.2264952788528545610e3)
_C = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
      3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
      1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
      1.05075007164441684324e-9)
_E = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
      2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
      7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
      2.04426310338993978564e-15)


def _poly(coef, x):
    v = 0.0
    for c in reversed(coef):
        v = v * x + c
    return v


def qnorm(p):
    """Normal quantile, Wichura's AS 241 (the algorithm R's qnorm uses)."""
    p = float(p)
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _poly(_A, r) / _poly(_B, r)
    r = p if q < 0 else 1.0 - p
    if r <= 0.0:
        return float("-inf") if q < 0 else float("inf")
    r = sqrt(-log(r))
    if r <= 5.0:
        val = _poly(_C, r - 1.6) / _poly(_D, r - 1.6)
    else:
        val = _poly(_E, r - 5.0) / _poly(_F, r - 5.0)
    return -val if q < 0 else val


def rnorm(n, seed=0, stream=0):
    """``n`` standard normals by inverse CDF -- one uniform per normal, so
    two implementations that share the uniform stream stay in step."""
    return [qnorm(u) for u in runif(n, seed=seed, stream=stream)]


def permutation(n, seed=0, stream=0):
    """A Fisher-Yates permutation of ``range(n)`` driven by ``runif``.

    Swaps run downward, i from n-1 to 1, consuming one uniform each, so
    the R mirror can reproduce it exactly by consuming the same stream
    in the same order.
    """
    idx = list(range(n))
    u = runif(max(n - 1, 0), seed=seed, stream=stream)
    for pos, i in enumerate(range(n - 1, 0, -1)):
        j = int(u[pos] * (i + 1))
        if j > i:
            j = i
        idx[i], idx[j] = idx[j], idx[i]
    return idx
