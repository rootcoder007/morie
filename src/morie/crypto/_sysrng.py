"""Operating-system randomness for the crypto package.

Key material must not come from a seeded simulation generator: morie's
``default_rng()`` without a seed is deterministic (seed 0), so every key
it produced was the same on every run. This generator draws from
``random.SystemRandom`` (os.urandom) and offers the few numpy-Generator
methods the crypto cores use.
"""

from __future__ import annotations

import random as _random

from morie.fn import _array_core as np

_SYS = _random.SystemRandom()


class SystemGenerator:
    """numpy-Generator-shaped facade over the OS CSPRNG."""

    def _shape(self, size, draw):
        if size is None:
            return draw()
        shape = (size,) if isinstance(size, int) else tuple(size)
        if len(shape) == 1:
            return np.array([draw() for _ in range(shape[0])])
        rows, cols = shape
        return np.array([[draw() for _ in range(cols)] for _ in range(rows)])

    def integers(self, low, high=None, size=None, dtype=None):
        if high is None:
            low, high = 0, low
        lo, hi = int(low), int(high)
        out = self._shape(size, lambda: lo + _SYS.randrange(hi - lo))
        return out.astype(dtype) if dtype is not None and size is not None else out

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._shape(size, lambda: _SYS.gauss(float(loc), float(scale)))

    def permutation(self, n):
        idx = list(range(int(n)))
        _SYS.shuffle(idx)
        return np.array(idx)

    def choice(self, a, size=None, replace=True):
        pool = list(range(int(a))) if isinstance(a, int) else list(a)
        if replace:
            return self._shape(size, lambda: _SYS.choice(pool))
        k = 1 if size is None else int(size)
        picks = _SYS.sample(pool, k)
        return picks[0] if size is None else np.array(picks)


def system_rng() -> SystemGenerator:
    return SystemGenerator()
