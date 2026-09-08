# morie.fn -- function file (rootcoder007/morie)
"""GP length-scale adaptation.

Implements sec. 11.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_adapt_thm"]


def _chol_solve(K, y):
    """Solve K x = y via the native linalg solver."""
    x = np.linalg.solve(np.marr(K), np.marr(y))
    return [float(v) for v in x._flat()]


def ghosal_gp_adapt_thm(n=60, l_true=0.2,
                        l_grid=(0.05, 0.2, 0.8), noise=0.1,
                        seed=42):
    """f ~ GP(0, k_l), l ~ Pi_l: the evidence-weighted posterior over
    the length scale concentrates near the scale of the truth,
    adapting the rate to unknown smoothness (sec. 11.6). Exact
    log marginal likelihood per candidate l. Keys: estimate."""
    rng = np.random.default_rng(seed)
    xs = [(i + 0.5) / n for i in range(n)]
    f0 = [math.sin(2.0 * math.pi * x / (5.0 * l_true)) for x in xs]
    ys = [f + noise * float(rng.normal(0, 1)) for f in f0]
    def logev(l):
        K = [[math.exp(-0.5 * ((xs[i] - xs[j]) / l) ** 2)
              + (noise ** 2 + 1e-8 if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        alpha = _chol_solve(K, ys)
        quad = sum(a * y for a, y in zip(alpha, ys))
        # log det via elimination
        m = [row[:] for row in K]
        ld = 0.0
        for i in range(n):
            ld += math.log(m[i][i])
            for r in range(i + 1, n):
                fmul = m[r][i] / m[i][i]
                for c in range(i, n):
                    m[r][c] -= fmul * m[i][c]
        return -0.5 * quad - 0.5 * ld
    evs = [logev(l) for l in l_grid]
    l_hat = l_grid[evs.index(max(evs))]
    res = RichResult(payload={"estimate": l_hat,
                              "log_evidence": evs,
                              "method": "GP length-scale adaptation (GvdV 2017 sec. 11.6)"})
    return with_describe_pointer(res, "gh_c11_13")


def cheatsheet():
    return "gh_c11_13: GP length-scale adaptation"
