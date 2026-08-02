# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoE auxiliary load-balancing loss (Shazeer / Switch Transformer)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_moe_load_balance_loss"]


def kamath_moe_load_balance_loss(fractions, gate_means, N, alpha, tol=1e-6):
    """L_aux = alpha * N * sum_i f_i * P_i.

    ``f_i`` is the fraction of tokens dispatched to expert i (sums to
    1) and ``P_i`` the mean router probability for expert i (sums to
    1). Perfect balance gives f_i = P_i = 1/N and L_aux = alpha; the
    payload reports that floor next to the value so a caller can see
    how far from balanced the router is, instead of staring at a bare
    number.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 2,
    auxiliary load-balancing loss; that section is not in the 2024
    PDF, so the loss is implemented exactly as the spec line states
    (Fedus et al. 2021, Eq 4).

    Examples
    --------
    >>> out = kamath_moe_load_balance_loss([0.5, 0.5], [0.5, 0.5], 2, 0.01)
    >>> abs(out["estimate"] - 0.01) < 1e-15
    True
    >>> out["balanced_floor"]
    0.01
    >>> bad = kamath_moe_load_balance_loss([1.0, 0.0], [1.0, 0.0], 2, 0.01)
    >>> abs(bad["estimate"] - 0.02) < 1e-15
    True
    """
    f = np.atleast_1d(np.asarray(fractions, dtype=float)).ravel()
    P = np.atleast_1d(np.asarray(gate_means, dtype=float)).ravel()
    N = int(N)
    alpha = float(alpha)
    if N < 1:
        raise ValueError(f"N must be at least 1; got {N}.")
    if f.size != N or P.size != N:
        raise ValueError(
            f"expected {N} experts; got {f.size} fractions and "
            f"{P.size} gate means.")
    if np.any(f < 0) or np.any(P < 0):
        raise ValueError("fractions and gate means must be non-negative.")
    if abs(f.sum() - 1.0) > tol:
        raise ValueError(
            f"the dispatch fractions sum to {f.sum():.6f}, not 1; every "
            "token goes somewhere.")
    if abs(P.sum() - 1.0) > tol:
        raise ValueError(
            f"the mean router probabilities sum to {P.sum():.6f}, not 1.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    val = alpha * N * float(np.dot(f, P))
    return RichResult(payload={
        "estimate": val, "loss": val,
        "per_expert": [float(a * b) for a, b in zip(f, P)],
        "balanced_floor": alpha,
        "imbalance_ratio": (val / alpha) if alpha > 0 else float("nan"),
        "alpha": alpha, "n": N,
        "method": "MoE auxiliary load-balancing loss alpha*N*sum f_i P_i"})


def cheatsheet():
    return "kmlb: alpha*N*sum(f_i P_i); equals alpha iff perfectly balanced"
