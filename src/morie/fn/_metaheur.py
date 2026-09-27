"""Shared machinery for the population metaheuristics: Philox draws, bounds, bookkeeping.

Draws come from buffered Philox streams (uniforms on odd streams 1, 3, 5, ...,
normals on even streams 2, 4, ...; 1024 per block), consumed one at a time in
the same order by both language arms, so a seed reproduces a run exactly.
"""

from ._rng import random_normal, random_uniform

BLOCK = 1024


class Rand:
    def __init__(self, seed):
        self.seed = int(seed)
        self.ub, self.ui, self.us = [], 0, -1
        self.nb, self.ni, self.ns = [], 0, 0

    def u(self):
        if self.ui >= len(self.ub):
            self.us += 2
            self.ub = [float(v) for v in random_uniform(BLOCK, seed=self.seed, stream=self.us)]
            self.ui = 0
        self.ui += 1
        return self.ub[self.ui - 1]

    def n(self):
        if self.ni >= len(self.nb):
            self.ns += 2
            self.nb = [float(v) for v in random_normal(BLOCK, seed=self.seed, stream=self.ns)]
            self.ni = 0
        self.ni += 1
        return self.nb[self.ni - 1]

    def idx(self, k):
        """Uniform integer in 0 .. k - 1."""
        return min(int(self.u() * k), k - 1)

    def other(self, k, i):
        """Uniform integer in 0 .. k - 1 other than i."""
        j = self.idx(k - 1)
        return j + 1 if j >= i else j


def setup(f, bounds, n_pop, rnd):
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    if not lo or any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("bounds must be (low, high) pairs with low < high")
    if int(n_pop) < 2:
        raise ValueError("n_pop must be at least 2")
    X = [[lo[j] + rnd.u() * (hi[j] - lo[j]) for j in range(len(lo))] for _ in range(int(n_pop))]
    F = [float(f(x)) for x in X]
    return lo, hi, X, F


def clip(x, lo, hi):
    return [min(max(v, a), b) for v, a, b in zip(x, lo, hi)]


def argmin(F):
    b = 0
    for i in range(1, len(F)):
        if F[i] < F[b]:
            b = i
    return b


# Mantegna sigma_u for beta = 3/2, (Gamma(5/2) sin(3 pi / 4) / (Gamma(5/4) (3/2) 2^(1/4)))^(2/3), as a literal so both arms agree to the bit
LEVY_S = 0.6965745025576967


def levy(rnd, d):
    """Mantegna (1994) Levy step with beta = 3/2: u / |v|^(2/3), u ~ N(0, LEVY_S^2), v ~ N(0, 1)."""
    out = []
    for _ in range(d):
        u = rnd.n() * LEVY_S
        v = rnd.n()
        out.append(u / abs(v) ** (1 / 1.5))
    return out


def result(title, method, x, fx, hist, nfev):
    from ._richresult import RichResult

    return RichResult(
        title=title,
        summary_lines=[("best f", fx), ("evaluations", nfev)],
        payload={"x": list(x), "fun": fx, "estimate": fx, "history": hist, "n_fev": nfev, "method": method},
    )
