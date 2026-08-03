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
itself is Matsumoto and Nishimura (1998).

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
