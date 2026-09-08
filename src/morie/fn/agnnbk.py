# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero residual block.

Source consulted: He, K., Zhang, X., Ren, S. and Sun, J. (2016).  Deep
residual learning for image recognition.  *CVPR*, 770-778
(arXiv:1512.03385), equation (1), y = F(x, {W_i}) + x; and Silver et al.
(2017), *Nature* 550, 354-359, methods, which specialises it to a block
of two 3x3 convolutions each followed by batch normalisation, with a
rectifier after the first and after the skip addition:

    y = relu( x + BN(conv2( relu( BN(conv1(x)) ) )) )

Silver et al. (2018), arXiv:1712.01815 (FETCHED), uses the same tower.

Batch normalisation here is the inference-time affine using the
statistics of the input itself (Ioffe and Szegedy 2015, arXiv:1502.03167,
equation 3): xhat = (x - mean) / sqrt(var + eps).  Convolution is 1-D
and 'same'-padded with zeros; ``filters`` is either the list of kernels
or an integer, in which case the identity kernel [1.0] is repeated that
many times -- the honest degenerate case, which computes the block
without inventing any weights.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_resnet_block"]

_BN_EPS = 1e-5


def _bn(v):
    m = k.mean(v)
    s = 0.0
    for x in v:
        s += (x - m) * (x - m)
    var = s / len(v) if v else 0.0
    d = math.sqrt(var + _BN_EPS)
    return [(x - m) / d for x in v]


def _conv1d(v, kern):
    n = len(v)
    K = len(kern)
    off = K // 2
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(K):
            t = i + j - off
            if 0 <= t < n:
                s += kern[j] * v[t]
        out[i] = s
    return out


def alphazero_resnet_block(x, filters=1):
    """One pre-activation-free residual block over a 1-D feature vector.

    Parameters
    ----------
    x : array-like
        Input features.
    filters : int or list of list
        Convolution kernels for the two convolutions.  An integer n means
        n copies of the identity kernel [1.0]; a list of two kernels is
        used as conv1 and conv2.

    Returns
    -------
    RichResult with payload:
        estimate : the mean of the block output
        y        : the block output
        h1       : the intermediate activation after the first conv+BN+relu
    """
    v = k.vec(x)
    if isinstance(filters, (int, float)):
        kerns = [[1.0], [1.0]]
    else:
        kerns = [k.vec(filters[0]), k.vec(filters[1] if len(filters) > 1 else filters[0])]
    h = _conv1d(v, kerns[0])
    h = _bn(h)
    h1 = [z if z > 0.0 else 0.0 for z in h]
    h2 = _conv1d(h1, kerns[1])
    h2 = _bn(h2)
    y = [z if z > 0.0 else 0.0 for z in [v[i] + h2[i] for i in range(len(v))]]
    return RichResult(
        title="AlphaZero residual block",
        summary_lines=[("units", len(v))],
        payload={
            "estimate": k.mean(y),
            "y": y,
            "h1": h1,
            "n": len(v),
            "method": "Residual block y = relu(x + BN(conv2(relu(BN(conv1(x))))))",
        },
    )


def cheatsheet():
    return "agnnbk: AlphaZero residual network block"
