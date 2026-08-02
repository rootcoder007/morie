# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SwiGLU: the Swish-gated linear unit used in LLaMA-style FFNs."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_swiglu_activation", "swish"]


def swish(z, beta=1.0):
    """Swish(z) = z * sigma(beta z), computed branch-stably so a large
    negative z underflows to 0 instead of overflowing exp."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = (beta * z) >= 0
    out[pos] = z[pos] / (1.0 + np.exp(-beta * z[pos]))
    e = np.exp(beta * z[~pos])
    out[~pos] = z[~pos] * e / (1.0 + e)
    return out


def kamath_swiglu_activation(x, W, V, b=None, c=None):
    """SwiGLU(x, W, V, b, c) = Swish(x W + b) * (x V + c).

    Two projections, one of them gating the other elementwise -- so a
    SwiGLU FFN with the same parameter count as a ReLU one uses a
    smaller hidden width (the 2/3 factor in LLaMA). W and V must have
    the same shape, since the product is elementwise; that is checked
    rather than left to broadcast into something plausible-looking.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, SwiGLU
    (Shazeer 2020).

    Examples
    --------
    >>> out = kamath_swiglu_activation([1.0], [[1.0]], [[2.0]])
    >>> import math
    >>> abs(out["output"][0] - (1 / (1 + math.exp(-1.0))) * 2.0) < 1e-12
    True
    >>> zero = kamath_swiglu_activation([0.0], [[1.0]], [[2.0]])
    >>> zero["output"]
    [0.0]
    >>> shifted = kamath_swiglu_activation([0.0], [[1.0]], [[1.0]],
    ...                                    b=[0.0], c=[3.0])
    >>> shifted["output"]
    [0.0]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    W = np.atleast_2d(np.asarray(W, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if W.shape != V.shape:
        raise ValueError(
            f"W is {W.shape} and V is {V.shape}; the gate multiplies "
            "elementwise, so they must match.")
    if W.shape[0] != x.size:
        raise ValueError(
            f"x has {x.size} features but W expects {W.shape[0]}.")
    gate_pre = x @ W
    up_pre = x @ V
    if b is not None:
        bb = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
        if bb.size != gate_pre.size:
            raise ValueError(
                f"b has {bb.size} entries for a {gate_pre.size}-wide "
                "hidden layer.")
        gate_pre = gate_pre + bb
    if c is not None:
        cc = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
        if cc.size != up_pre.size:
            raise ValueError(
                f"c has {cc.size} entries for a {up_pre.size}-wide "
                "hidden layer.")
        up_pre = up_pre + cc
    g = swish(gate_pre)
    out = g * up_pre
    return RichResult(payload={
        "output": [float(v) for v in out],
        "gate": [float(v) for v in g],
        "linear": [float(v) for v in up_pre],
        "estimate": float(out[0]),
        "hidden_dim": int(out.size), "n": int(out.size),
        "method": "SwiGLU: Swish(xW + b) * (xV + c)"})


def cheatsheet():
    return "kmswig: Swish(xW+b) gates (xV+c) elementwise"
