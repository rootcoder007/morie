# morie.fn -- function file (rootcoder007/morie)
"""Whittle-likelihood spectral rate.

Implements sec. 9.5.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_whittle_crt"]


def ghosal_whittle_crt(ns=(256, 1024, 4096), n_bins=6, seed=42):
    """Spectral density estimation under the Whittle likelihood:
    rate n^{-s/(2s+1)} (log n)^{1/2} for s-smooth spectra
    (sec. 9.5.2). Binned exponential-likelihood posterior means for
    white noise contract to the flat spectrum. Keys: estimate."""
    rng = np.random.default_rng(seed)
    truth = 1.0 / (2.0 * math.pi)
    errs = []
    for n in ns:
        x = [float(rng.normal(0, 1)) for _ in range(n)]
        m = n // 2
        bs = [0.0] * n_bins
        bc = [0] * n_bins
        for j in range(1, m, max(1, m // 200)):    # subsample freqs
            ang = 2.0 * math.pi * j / n
            wr = sum(v * math.cos(ang * t) for t, v in enumerate(x))
            wi = sum(v * math.sin(ang * t) for t, v in enumerate(x))
            I = (wr * wr + wi * wi) / (2.0 * math.pi * n)
            b = min(int(n_bins * j / m), n_bins - 1)
            bs[b] += I
            bc[b] += 1
        est = [(0.5 * truth + s) / (0.5 + c)
               for s, c in zip(bs, bc)]
        errs.append(sum(abs(e - truth) for e in est) / n_bins)
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "Whittle spectral rate (GvdV 2017 sec. 9.5.2)"})
    return with_describe_pointer(res, "gh_c9_7")


def cheatsheet():
    return "gh_c9_7: Whittle-likelihood spectral rate"
