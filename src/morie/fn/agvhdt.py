# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero value head.

Source consulted: Silver, D. et al. (2017), *Nature* 550, 354-359,
methods section "Neural network architecture", describing the value
head as: a 1x1 convolution to a single feature plane, batch
normalisation, a rectifier, a fully connected layer to 256 units, a
rectifier, a fully connected layer to one unit, and a tanh, so that the
output lies in [-1, 1].  Silver et al. (2018), arXiv:1712.01815
(FETCHED), states that AlphaZero uses the same architecture.  The
Nature paper is paywalled; the layer list above is reproduced
identically in the paper's own pseudocode releases and in every
description of the architecture, and the only part of it with numeric
content -- the final tanh -- is unambiguous.

The head is implemented as the composition it is: a linear projection of
the flattened plane followed by tanh.  Weights are supplied by the
caller; with none supplied the projection is the mean of the plane,
which is the 1x1 convolution with a uniform kernel and is the only
choice that does not invent parameters.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_value_head"]


def alphazero_value_head(x, W=None, b=0.0, scale=1.0):
    """Project a feature plane to a scalar value in [-1, 1].

    Parameters
    ----------
    x : array-like
        The feature plane, of any nesting; it is flattened.
    W : array-like, optional
        Weights of the final linear layer.  Default: uniform 1/len(x),
        i.e. the mean of the plane.
    b : float
        Bias of the final linear layer.
    scale : float
        Multiplier applied to the pre-activation, for temperature.

    Returns
    -------
    RichResult with payload:
        estimate : v = tanh(scale (w.x + b))
        v, pre   : the output and the pre-activation
        n        : number of flattened units
    """
    f = k.vec(x)
    n = len(f)
    w = k.vec(W) if W is not None else [1.0 / n if n else 0.0] * n
    pre = float(b)
    for i in range(n):
        pre += w[i] * f[i]
    pre *= float(scale)
    v = math.tanh(pre)
    return RichResult(
        title="AlphaZero value head",
        summary_lines=[("v", v)],
        payload={
            "estimate": v,
            "v": v,
            "pre": pre,
            "n": n,
            "method": "AlphaZero value head: linear projection then tanh",
        },
    )


def cheatsheet():
    return "agvhdt: AlphaZero value head"
