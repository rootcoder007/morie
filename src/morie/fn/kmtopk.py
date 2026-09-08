# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Top-k expert gating: keep the k largest gate scores, renormalise,
zero the rest."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_moe_top_k_gating"]


def kamath_moe_top_k_gating(gates, k):
    """TopK(g)_i = g_i if i in argtopk(g, k) else 0;
    g' = TopK(g) / sum(TopK(g)).

    This renormalises ALREADY-COMPUTED gate scores. It is not the same
    as ``morie.fn.km040``, which masks the scores to -inf and takes a
    softmax over the survivors; softmax-after-masking and
    normalise-after-masking give different weights from the same
    scores, and conflating them is a real and silent bug. Use km040
    when starting from router logits, this when starting from gates
    that are already non-negative weights.

    Ties at the k-th place are broken by the lower index, so exactly k
    experts are kept and the choice is reproducible.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, top-k expert
    gating.

    Examples
    --------
    >>> out = kamath_moe_top_k_gating([0.5, 0.3, 0.2], 2)
    >>> [round(v, 12) for v in out["weights"]]
    [0.625, 0.375, 0.0]
    >>> out["selected_experts"]
    [0, 1]
    >>> abs(sum(out["weights"]) - 1.0) < 1e-15
    True
    """
    g = np.atleast_1d(np.asarray(gates, dtype=float)).ravel()
    k = int(k)
    n = g.size
    if n == 0:
        raise ValueError("no gate scores supplied.")
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}]; got {k}.")
    if not np.all(np.isfinite(g)):
        raise ValueError("gate scores must be finite.")
    if np.any(g < 0):
        raise ValueError(
            "gate scores must be non-negative; renormalising a negative "
            "weight can produce a mixture that is not a convex "
            "combination. Use km040 if these are router logits.")
    order = np.argsort(-g, kind="stable")[:k]
    kept = np.zeros(n)
    kept[order] = g[order]
    total = kept.sum()
    if total == 0:
        raise ValueError(
            "every selected gate is 0, so the renormalisation is 0/0 "
            "and no expert is chosen.")
    w = kept / total
    return RichResult(payload={
        "weights": [float(v) for v in w],
        "selected_experts": [int(i) for i in sorted(order)],
        "kept_mass": float(total / g.sum()) if g.sum() > 0 else 0.0,
        "n_active": int(np.sum(w > 0)),
        "estimate": float(w.max()),
        "k": k, "n": n,
        "method": "Top-k gate renormalisation (not a masked softmax)"})


def cheatsheet():
    return "kmtopk: keep top-k gates, renormalise; km040 does masked softmax"
