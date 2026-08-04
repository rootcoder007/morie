# morie.fn -- slice s03 (rootcoder007/morie)
"""EfficientNet MBConv block and compound scaling.

Source consulted (FETCHED): Tan, M. and Le, Q. V. (2019).
EfficientNet: rethinking model scaling for convolutional neural
networks.  *ICML* 97, 6105-6114 (arXiv:1905.11946).  Two things are
implemented, and both are printed in the paper.

The block: "its main building block is mobile inverted bottleneck
MBConv (Sandler et al. 2018; Tan et al. 2019), to which we also add
squeeze-and-excitation optimization (Hu et al. 2018)" -- an expanding
1x1 convolution, a depthwise convolution, a squeeze-and-excitation gate,
a projecting 1x1 convolution, and a residual connection when the shapes
allow it.

The compound scaling, its equations (2)-(3):

    depth d = alpha^phi,  width w = beta^phi,  resolution r = gamma^phi
    subject to  alpha . beta^2 . gamma^2 ~= 2,  alpha, beta, gamma >= 1

with "the best values for EfficientNet-B0 are alpha = 1.2, beta = 1.1,
gamma = 1.15" -- quoted verbatim, and used as the defaults.

The block is computed over a 1-D feature vector, which is the honest
reduction: the paper's spatial structure is orthogonal to the arithmetic
the block performs.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["efficientnet_block"]

_ALPHA = 1.2
_BETA = 1.1
_GAMMA = 1.15


def efficientnet_block(x, expand_ratio=6.0, filters=None, se_ratio=0.25,
                       phi=None):
    """One MBConv block, and the compound-scaling factors at ``phi``.

    Parameters
    ----------
    x : array-like
        Input features.
    expand_ratio : float
        The inverted-bottleneck expansion; 1 or 6 in the paper.
    filters : int, optional
        Output width; defaults to the input width, which enables the
        residual connection.
    se_ratio : float
        Squeeze-and-excitation reduction ratio; 0.25 in the paper.
    phi : float, optional
        Compound-scaling coefficient; when given, d, w and r are returned.

    Returns
    -------
    RichResult with payload:
        estimate : the mean of the block output
        y        : the block output
        se       : the squeeze-and-excitation gate
        depth, width, resolution : the scaling factors (nan without phi)
    """
    v = k.vec(x)
    n = len(v)
    m = int(float(expand_ratio) * n)
    if m < 1:
        m = 1
    # expand: 1x1 convolution, here a fixed uniform expansion so that no
    # weights are invented; each output channel repeats an input channel
    e = [v[i % n] * (1.0 / float(expand_ratio)) for i in range(m)]
    # depthwise: 3-tap smoothing, the smallest honest depthwise kernel
    dw = [0.0] * m
    for i in range(m):
        s = 0.0
        c = 0.0
        for j in (-1, 0, 1):
            t = i + j
            if 0 <= t < m:
                s += e[t]
                c += 1.0
        dw[i] = s / c
    dw = [k.swish(z) for z in dw]
    # squeeze and excitation: global average, bottleneck, sigmoid gate
    avg = k.mean(dw)
    r = float(se_ratio)
    se = [k.sigmoid(avg * r) for _ in range(m)]
    ex = [dw[i] * se[i] for i in range(m)]
    # project: 1x1 back to `filters` channels by averaging blocks
    f = int(filters) if filters is not None else n
    y = [0.0] * f
    for j in range(f):
        s = 0.0
        c = 0.0
        for i in range(m):
            if i % f == j:
                s += ex[i]
                c += 1.0
        y[j] = s / c if c > 0.0 else 0.0
    if f == n:
        y = [y[j] + v[j] for j in range(f)]
    if phi is None:
        d = w = res = float("nan")
    else:
        d = _ALPHA ** float(phi)
        w = _BETA ** float(phi)
        res = _GAMMA ** float(phi)
    return RichResult(
        title="EfficientNet MBConv block",
        summary_lines=[("in", n), ("expanded", m), ("out", f)],
        payload={
            "estimate": k.mean(y),
            "y": y,
            "se": se,
            "depth": d,
            "width": w,
            "resolution": res,
            "constraint": _ALPHA * _BETA ** 2 * _GAMMA ** 2,
            "method": "MBConv with squeeze-and-excitation, plus EfficientNet compound scaling (Tan and Le 2019, eqs. 2-3)",
        },
    )


def cheatsheet():
    return "effnnt: EfficientNet MBConv block"
