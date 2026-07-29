# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.2: the PARALLEL adapter update."""

import numpy as np

from ._richresult import RichResult
from .km054 import _adapter_core, _relu

__all__ = ["kamath_ch4_parallel_adapter"]


def kamath_ch4_parallel_adapter(H_o, H_i, W_down, W_up, f=None):
    """H_o <- H_o + f(H_i W_down) W_up.

    The parallel adapter reads the layer's INPUT H_i, so its branch
    does not depend on the frozen layer's output and needs no
    backpropagation through it. That single substitution is the whole
    difference from Eq 4.1, so the arithmetic is km054's, delegated.
    Passing H_i = H_o reproduces the series adapter exactly.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.2, printed
    p. 149.

    Examples
    --------
    >>> out = kamath_ch4_parallel_adapter([[1.0, 2.0]], [[0.0, 1.0]],
    ...                                   [[1.0], [0.0]], [[1.0, 1.0]])
    >>> out["output"]
    [[1.0, 2.0]]
    >>> out["delta"]
    [[0.0, 0.0]]
    """
    out, delta, r = _adapter_core(H_o, H_i, W_down, W_up,
                                  _relu if f is None else f)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "delta": [[float(v) for v in row] for row in delta],
        "bottleneck_rank": int(r), "estimate": float(out[0, 0]),
        "n": int(out.shape[0]),
        "method": "parallel adapter (Kamath Eq 4.2)"})


def cheatsheet():
    return "km055: H_o + f(H_i W_down) W_up, bottleneck fed by the input"
