# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.41: the Mixtral block -- top-2 gating over SwiGLU
experts."""

import numpy as np

from ._richresult import RichResult
from .km039 import kamath_ch2_moe_output
from .km040 import kamath_ch2_moe_topk_gating

__all__ = ["kamath_ch2_mixtral_swiglu_moe"]


def _swiglu(x, W1, W3, W2):
    a = x @ W1
    swish = a / (1.0 + np.exp(-a))
    return (swish * (x @ W3)) @ W2


def kamath_ch2_mixtral_swiglu_moe(x, W_g, expert_weights=None):
    """y = sum_i softmax(Top2(x W_g))_i SwiGLU_i(x), composed from
    Eq 2.40's gate and Eq 2.39's combination so the three formulas
    cannot drift. ``expert_weights`` is a list of (W1, W3, W2) per
    expert; SwiGLU(x) = (swish(x W1) * (x W3)) W2 (Shazeer 2020).
    Without expert weights each expert defaults to the identity, which
    isolates the gate for testing.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.41, printed
    p. 75.

    Examples
    --------
    >>> out = kamath_ch2_mixtral_swiglu_moe([1.0], [[3.0, 1.0, 2.0]])
    >>> out["gate"]["n_active"]
    2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    gate = kamath_ch2_moe_topk_gating(x, W_g, k=2)
    n = len(gate["weights"])
    if expert_weights is None:
        experts = [(lambda xv: xv) for _ in range(n)]
    else:
        if len(expert_weights) != n:
            raise ValueError(
                f"need one (W1, W3, W2) triple per expert; got "
                f"{len(expert_weights)} for {n}.")
        def make(ws):
            W1 = np.atleast_2d(np.asarray(ws[0], dtype=float))
            W3 = np.atleast_2d(np.asarray(ws[1], dtype=float))
            W2 = np.atleast_2d(np.asarray(ws[2], dtype=float))
            if W1.shape != W3.shape:
                raise ValueError("W1 and W3 must share a shape.")
            if W1.shape[1] != W2.shape[0]:
                raise ValueError("W2's rows must match W1's columns.")
            return lambda xv: _swiglu(xv, W1, W3, W2)
        experts = [make(ws) for ws in expert_weights]
    combined = kamath_ch2_moe_output(x, gate["weights"], experts)
    return RichResult(payload={
        "output": combined["output"], "gate": gate.payload,
        "experts_evaluated": combined["experts_evaluated"],
        "estimate": combined["estimate"], "n": n,
        "method": "Mixtral top-2 SwiGLU MoE (Kamath Eq 2.41)"})


def cheatsheet():
    return "km041: Eq 2.40 gate + Eq 2.39 combine over SwiGLU experts"
