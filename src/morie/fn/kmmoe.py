# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoE router: softmax-gated top-k expert selection and combination."""

import numpy as np

from ._richresult import RichResult
from .km039 import kamath_ch2_moe_output
from .km040 import kamath_ch2_moe_topk_gating

__all__ = ["kamath_moe_router_softmax"]


def kamath_moe_router_softmax(x, Wr, experts, k):
    """g(x) = softmax(W_r x);  y = sum_{i in TopK(g)} g_i Expert_i(x).

    Both halves already exist in this package, so both are DELEGATED:
    the gate to ``morie.fn.km040`` (Kamath Eq 2.40, softmax over the
    top-k scores with the rest exactly 0) and the weighted sum to
    ``morie.fn.km039`` (Eq 2.39, which skips zero-weight experts so
    the unselected ones never run). Two copies of one formula drift.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, mixture-of-experts
    (Eq 2.39-2.40, Mixtral).

    Examples
    --------
    >>> out = kamath_moe_router_softmax([1.0], [[3.0, 1.0, 2.0]],
    ...     [lambda v: v[0] * 2, lambda v: v[0] * 4, lambda v: v[0] * 8],
    ...     k=2)
    >>> out["n_active"]
    2
    >>> import math
    >>> w0 = math.exp(3) / (math.exp(3) + math.exp(2))
    >>> abs(out["gate_weights"][0] - w0) < 1e-12
    True
    >>> abs(out["estimate"] - (w0 * 2 + (1 - w0) * 8)) < 1e-12
    True
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    W = np.atleast_2d(np.asarray(Wr, dtype=float))
    experts = list(experts)
    if W.shape[1] != len(experts):
        raise ValueError(
            f"the router scores {W.shape[1]} experts but "
            f"{len(experts)} were supplied.")
    gate = kamath_ch2_moe_topk_gating(x, W, k=int(k))
    combined = kamath_ch2_moe_output(x, gate["weights"], experts)
    return RichResult(payload={
        "output": combined["output"],
        "gate_weights": gate["weights"],
        "selected_experts": gate["selected_experts"],
        "n_active": gate["n_active"],
        "experts_evaluated": combined["experts_evaluated"],
        "estimate": combined["estimate"],
        "k": int(k), "n": len(experts),
        "method": "Softmax top-k MoE router (delegates to km040 + km039)"})


def cheatsheet():
    return "kmmoe: km040 gate + km039 combine; unselected experts never run"
