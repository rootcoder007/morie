# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 2: per-expert capacity under a capacity factor."""

import math

from ._richresult import RichResult

__all__ = ["kamath_expert_capacity_factor"]


def kamath_expert_capacity_factor(tokens_per_batch, num_experts, C):
    r"""capacity = C * (tokens_per_batch / num_experts); overflow drops.

    ``capacity`` is the exact real-valued buffer size and ``slots`` is
    what an implementation allocates (the ceiling). ``C = 1`` is the
    perfectly balanced case; the extra headroom C - 1 is reported as
    the number of surplus token slots per expert, and the total slots
    across experts tell you how many tokens can be routed at all
    before any are dropped.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Capacity Factor;
    Fedus et al. (2022).

    Examples
    --------
    >>> out = kamath_expert_capacity_factor(100, 4, 1.25)
    >>> out["capacity"], out["slots"]
    (31.25, 32)
    >>> out["min_dropped"]        # 4*32 = 128 slots for 100 tokens
    0
    """
    t = float(tokens_per_batch)
    e = int(num_experts)
    c = float(C)
    if t <= 0:
        raise ValueError("a batch with no tokens has no capacity to "
                         "allocate.")
    if e < 1:
        raise ValueError(f"there must be at least one expert; got {e}.")
    if c <= 0:
        raise ValueError(f"the capacity factor must be positive; got "
                         f"{c}.")
    cap = c * (t / e)
    slots = int(math.ceil(cap))
    return RichResult(payload={
        "estimate": cap, "capacity": cap, "slots": slots,
        "total_slots": slots * e,
        "min_dropped": max(0, int(math.ceil(t)) - slots * e),
        "headroom_per_expert": cap - t / e,
        "num_experts": e, "n": int(math.ceil(t)),
        "method": "MoE per-expert capacity (Kamath Ch 2)"})


def cheatsheet():
    return "kmcap: C * tokens/experts, with the drop threshold reported"
