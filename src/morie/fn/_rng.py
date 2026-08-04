# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native random number generator: Philox4x32-10 and Wichura's AS 241.

Two published algorithms, implemented rather than delegated, so that the R
and Python arms of morie draw the SAME numbers rather than merely
statistically similar ones.

**Uniforms -- Philox4x32-10.** Salmon, Moraes, Dror and Shaw (2011),
"Parallel Random Numbers: As Easy as 1, 2, 3", Proc. SC'11. A counter-based
generator: the n-th output is a keyed bijection of the integer n, so there is
no evolving state to keep in step, streams may be indexed at any offset, and
the arithmetic is integer-only -- which is what makes it reproduce bit for
bit across languages and platforms. Period 2^130 per key; passes BigCrush.

This replaces the 32-bit linear congruential generator that had been copied
between the `gr*` modules,

    s = (1664525 s + 1013904223) mod 2^32,

whose period of 2^32 is exhausted by a few billion draws, and whose tuples
fall on a small number of lattice hyperplanes (Marsaglia, 1968) -- the
failure mode that matters most for simulating spatial fields, since those are
built from exactly such tuples.

**Normals -- Wichura's AS 241.** Wichura (1988), "Algorithm AS 241: The
Percentage Points of the Normal Distribution", Applied Statistics 37(3),
477-484: the rational approximation to the normal quantile function that R's
`qnorm` uses, accurate to about 1e-16 over the whole range. Normals are drawn
by the inverse CDF rather than by Box-Muller, for two reasons: it consumes
exactly one uniform per normal, so two implementations stay in step, and it
avoids `log` and `cos`, whose last-bit behaviour differs between platforms'
libm.

Correctness is checked against the published Known Answer Tests rather than
asserted -- see the test suite.
"""

from . import _array_core as np

__all__ = []

_MASK32 = 0xFFFFFFFF
# Philox multipliers and Weyl constants, Salmon et al. (2011) Table 1.
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9          # golden ratio
_PHILOX_W1 = 0xBB67AE85          # sqrt(3) - 1
_PHILOX_ROUNDS = 10


def philox4x32(counter, key, rounds=_PHILOX_ROUNDS):
    """The Philox4x32 bijection: four 32-bit words in, four out.

    `counter` is either ONE block of four 32-bit words -- shape (4,) -- or a
    sequence of such blocks, shape (n, 4); `key` is a pair of 32-bit words.
    The result matches the argument: a 4-tuple for one block, a list of
    4-tuples for a sequence of them.

    Plain Python integers throughout. They are arbitrary precision, so the
    32x32 -> 64 bit products are exact under explicit masking and no unsigned
    array dtype is needed -- which is what lets this reproduce the R arm bit
    for bit. (An array backend would be actively wrong here: `&` on the
    array type is boolean, not bitwise, so masking a counter word would
    silently collapse it to 0 or 1 and Philox would emit plausible garbage.)
    """
    one = len(counter) == 4 and not hasattr(counter[0], "__len__")
    rows = (counter,) if one else counter
    key0 = int(key[0]) & _MASK32
    key1 = int(key[1]) & _MASK32
    out = []
    for row in rows:
        c0, c1, c2, c3 = (int(v) & _MASK32 for v in row)
        k0, k1 = key0, key1
        for r in range(rounds):
            p0 = _PHILOX_M0 * c0
            p1 = _PHILOX_M1 * c2
            c0, c1, c2, c3 = (((p1 >> 32) ^ c1 ^ k0) & _MASK32, p1 & _MASK32,
                              ((p0 >> 32) ^ c3 ^ k1) & _MASK32, p0 & _MASK32)
            if r + 1 < rounds:                # bump the key between rounds
                k0 = (k0 + _PHILOX_W0) & _MASK32
                k1 = (k1 + _PHILOX_W1) & _MASK32
        out.append((c0, c1, c2, c3))
    return out[0] if one else out


def random_uniform(n, seed=0, stream=0):
    """`n` uniforms in the open interval (0, 1).

    Word i of counter block j gives one uniform. The open interval matters:
    a normal quantile at u = 0 or u = 1 is infinite, so the transformation
    (w + 0.5) / 2^32 is used, which can reach neither endpoint.
    """
    n = int(n)
    if n < 0:
        raise ValueError("`n` must be non-negative")
    if n == 0:
        return np.empty(0)
    key = (int(seed) & _MASK32, (int(seed) >> 32) & _MASK32)
    stream = int(stream) & _MASK32
    words = []
    for j in range((n + 3) // 4):
        words.extend(philox4x32((j & _MASK32, (j >> 32) & _MASK32, stream, 0), key))
    return np.array([(w + 0.5) / 4294967296.0 for w in words[:n]], dtype=float)


# --- Wichura (1988) AS 241, the PPND16 coefficients -----------------------
_A = (3.3871328727963666080e0, 1.3314166789178437745e2,
      1.9715909503065514427e3, 1.3731693765509461125e4,
      4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3)
_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2,
      5.3941960214247511077e3, 2.1213794301586595867e4,
      3.9307895800092710610e4, 2.8729085735721942674e4,
      5.2264952788528545610e3)
_C = (1.42343711074968357734e0, 4.63033784615654529590e0,
      5.76949722146069140550e0, 3.64784832476320460504e0,
      1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
      6.89767334985100004550e-1, 1.48103976427480074590e-1,
      1.51986665636164571966e-2, 5.47593808499534494600e-4,
      1.05075007164441684324e-9)
_E = (6.65790464350110377720e0, 5.46378491116411436990e0,
      1.78482653991729133580e0, 2.96560571828504891230e-1,
      2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
      1.48753612908506148525e-2, 7.86869131145613259100e-4,
      1.84631831751005468180e-5, 1.42151175831644588870e-7,
      2.04426310338993978564e-15)


def _poly(coef, x):
    out = np.full_like(x, coef[-1])
    for c in coef[-2::-1]:
        out = out * x + c
    return out


def normal_quantile(p):
    """The standard normal quantile function, Wichura's AS 241 (PPND16).

    Split at |p - 1/2| <= 0.425 (the central branch), then at r <= 5 for the
    tails, exactly as the published algorithm prescribes.
    """
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("`p` must lie strictly inside (0, 1)")
    q = p - 0.5
    out = np.empty_like(p)

    central = np.abs(q) <= 0.425
    if np.any(central):
        r = 0.180625 - q[central] * q[central]
        out[central] = q[central] * _poly(_A, r) / _poly(_B, r)

    tail = ~central
    if np.any(tail):
        qt = q[tail]
        r = np.where(qt < 0.0, p[tail], 1.0 - p[tail])
        r = np.sqrt(-np.log(r))
        near = r <= 5.0
        rr = np.where(near, r - 1.6, r - 5.0)
        val = np.where(near,
                       _poly(_C, rr) / _poly(_D, rr),
                       _poly(_E, rr) / _poly(_F, rr))
        out[tail] = np.where(qt < 0.0, -val, val)
    return out


def random_normal(n, seed=0, stream=0):
    """`n` standard normals, one uniform each, by the inverse CDF."""
    return normal_quantile(random_uniform(n, seed=seed, stream=stream))


def random_multivariate_normal(mean, cov, seed=0, stream=0, jitter=1e-10):
    """One draw from N(mean, cov) by the Cholesky factor.

    Z = mean + L e with L L' = cov and e standard normal, which is the
    construction Schabenberger & Gotway use for simulating a Gaussian random
    field. `jitter` is added to the diagonal only if the factorisation fails,
    and the amount used is reported by the caller-visible exception if it
    still fails.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    n = mean.size
    if cov.shape != (n, n):
        raise ValueError("`cov` must be square and match `mean`")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + jitter * np.eye(n))
    return mean + chol @ random_normal(n, seed=seed, stream=stream)
