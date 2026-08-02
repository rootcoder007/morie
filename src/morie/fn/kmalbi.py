# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 2: ALiBi -- attention with linear distance biases."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["kamath_alibi_bias"]


def kamath_alibi_bias(Q, K, V, slopes, causal=False):
    r"""Attn = softmax(QK^T/sqrt(d_k) + m*D)V with D_ij = -(i - j).

    The bias is a plain ADDITIVE term on the scores, which is what
    ``morie.fn.attsdp`` already takes as its ``mask``; so the softmax
    and the value mix are delegated there and only the bias matrix is
    built here. ``slopes`` is one head-specific m (scalar) or a list
    of them, in which case one attention output per head is returned.
    ``causal=True`` additionally forbids attending to the future, as
    decoder-only ALiBi does.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, ALiBi; Press et al.
    (2022).

    Examples
    --------
    >>> out = kamath_alibi_bias([[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]], 0.0)
    >>> round(out["output"][0][0][0], 6)   # slope 0 -> plain attention
    0.669762
    >>> out["bias"][0][1]                  # D_01 = -(0 - 1) = 1
    1.0
    """
    Qm = np.atleast_2d(np.asarray(Q, dtype=float))
    Km = np.atleast_2d(np.asarray(K, dtype=float))
    m = np.atleast_1d(np.asarray(slopes, dtype=float))
    if m.size == 0:
        raise ValueError("no ALiBi slopes were given.")
    if np.any(m < 0):
        raise ValueError("ALiBi slopes are non-negative; a negative m "
                         "rewards distance instead of penalizing it.")
    i = np.arange(Qm.shape[0])[:, None]
    j = np.arange(Km.shape[0])[None, :]
    D = -(i - j).astype(float)
    outs, attns = [], []
    for slope in m:
        bias = slope * D
        if causal:
            bias = np.where(j <= i, bias, -np.inf)
        r = scaled_dot_product_attention(Qm, Km, V, mask=bias)
        outs.append(r["output"])
        attns.append(r["attention"])
    return RichResult(payload={
        "estimate": float(outs[0][0][0]), "output": outs,
        "attention": attns,
        "bias": [[float(v) for v in row] for row in D],
        "slopes": [float(v) for v in m], "n": int(Qm.shape[0]),
        "method": "ALiBi biased attention (Kamath Ch 2; softmax core "
                  "in attsdp)"})


def cheatsheet():
    return "kmalbi: attsdp with the additive -m*(i-j) distance bias"
