# morie.fn -- function file (rootcoder007/morie)
"""Time-series (Whittle) contraction.

Implements sec. 8.3.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ts_crt"]


def ghosal_ts_crt(ns=(256, 2048), n_bins=8, seed=42):
    """Spectral density via the Whittle likelihood: periodogram
    ordinates are approximately independent exponentials with mean
    f(omega); binned conjugate (inverse-gamma-type) posteriors for a
    flat truth contract as n grows (sec. 8.3.5). Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        x = [float(rng.normal(0, 1)) for _ in range(n)]
        m = n // 2
        binsum = [0.0] * n_bins
        bincnt = [0] * n_bins
        for j in range(1, m):
            wr = wi = 0.0
            ang = 2.0 * math.pi * j / n
            for t, v in enumerate(x):
                wr += v * math.cos(ang * t)
                wi += v * math.sin(ang * t)
            I = (wr * wr + wi * wi) / (2.0 * math.pi * n)
            b = min(int(n_bins * j / m), n_bins - 1)
            binsum[b] += I
            bincnt[b] += 1
        truth = 1.0 / (2.0 * math.pi)
        est = [(0.5 + s) / (1.0 + c) for s, c in zip(binsum, bincnt)]
        errs.append(sum(abs(e - truth) for e in est) / n_bins)
    res = RichResult(payload={"estimate": errs[-1],
                              "error_by_n": errs,
                              "contracting": errs[-1] < errs[0],
                              "method": "Whittle spectral contraction (GvdV 2017 sec. 8.3.5)"})
    return with_describe_pointer(res, "gh_c8_11")


def cheatsheet():
    return "gh_c8_11: Time-series (Whittle) contraction"


# compact alias per ledger/NAMING.md
ghosaltscrt = ghosal_ts_crt
