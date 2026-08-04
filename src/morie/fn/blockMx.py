# morie.fn -- function file (rootcoder007/morie)
"""Block-maxima method.

Implements sec. 3.1 & 3.3 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["block_maxima"]


def block_maxima(x, block_size):
    """Block-maxima GEV analysis (Coles 2001 sec. 3.1): split the
    series into blocks of ``block_size``, take each block maximum, fit
    the GEV by ML (sec. 3.3.2). ``estimate`` is the fitted location."""
    xs = _ev._flat(x)
    b = int(block_size)
    if b < 1 or len(xs) < 2 * b:
        raise ValueError("need at least two full blocks")
    maxima = [max(xs[i:i + b]) for i in range(0, len(xs) - b + 1, b)]
    f = _ev.gev_mle(maxima)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "sigma": f["sigma"], "xi": f["xi"],
                              "ll": f["loglik"], "n_blocks": len(maxima),
                              "maxima": maxima,
                              "method": "block maxima + GEV MLE (Coles 2001 sec. 3.1, 3.3)"})
    return with_describe_pointer(res, "blockMx")


def cheatsheet():
    return "blockMx: Block-maxima method"


# compact alias per ledger/NAMING.md
blockmaxima = block_maxima
