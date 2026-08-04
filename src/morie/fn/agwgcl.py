# morie.fn -- slice s03 (rootcoder007/morie)
"""Global-norm gradient clipping.

Source consulted (FETCHED): Pascanu, R., Mikolov, T. and Bengio, Y.
(2013).  On the difficulty of training recurrent neural networks.
*ICML* 28, 1310-1318 (arXiv:1211.5063), algorithm 1:

    if ||g|| >= threshold:  g <- threshold * g / ||g||

with ||.|| the L2 norm of the concatenation of every parameter's
gradient -- the *global* norm, not a per-tensor one.  AlphaZero itself
(Silver et al., arXiv:1712.01815 -- FETCHED) does not state a clipping
threshold; the routine is provided here as the standard stabiliser it
is, and the docstring says so rather than attributing it to AlphaZero.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_weight_clipping"]


def alphazero_weight_clipping(grad, max_norm=1.0):
    """Rescale a gradient so its global L2 norm is at most ``max_norm``.

    Parameters
    ----------
    grad : array-like
        The gradient, of any nesting; it is flattened to compute the norm.
    max_norm : float
        The clipping threshold.

    Returns
    -------
    RichResult with payload:
        estimate  : the norm after clipping
        clipped   : the rescaled gradient, flattened
        norm      : the norm before clipping
        scale     : the factor applied (1 when no clipping was needed)
        was_clipped
    """
    g = k.vec(grad)
    s2 = 0.0
    for x in g:
        s2 += x * x
    nrm = math.sqrt(s2)
    mx = float(max_norm)
    if nrm >= mx and nrm > 0.0:
        scale = mx / nrm
        was = True
    else:
        scale = 1.0
        was = False
    out = [x * scale for x in g]
    s2b = 0.0
    for x in out:
        s2b += x * x
    return RichResult(
        title="Gradient clipping",
        summary_lines=[("norm", nrm), ("clipped", was)],
        payload={
            "estimate": math.sqrt(s2b),
            "clipped": out,
            "norm": nrm,
            "scale": scale,
            "was_clipped": was,
            "method": "Global-norm gradient clipping (Pascanu et al. 2013, alg. 1)",
        },
    )


def cheatsheet():
    return "agwgcl: AlphaZero gradient clipping"
