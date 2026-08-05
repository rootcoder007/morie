# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Sequential composition of k differentially private mechanisms.

Dwork and Roth (2014), *The Algorithmic Foundations of Differential
Privacy*, Foundations and Trends in Theoretical Computer Science
9(3-4):211-407, doi:10.1561/0400000042, section 3.5, Theorem 3.16: if
M_i is (eps_i, delta_i)-differentially private for i in [k], then the
mechanism returning (M_1(x), ..., M_k(x)) is

    ( sum_i eps_i , sum_i delta_i )-differentially private.

"The epsilons and the deltas add up."  The budget consumed is therefore
the plain sum, whatever the mechanisms are, and the Laplace scale a
sensitivity-1 query can afford at step i is 1 / eps_i.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["k_step_dp_composition"]


def k_step_dp_composition(y, epsilons, deltas=None):
    """Total privacy budget spent by k mechanisms run in sequence.

    Parameters
    ----------
    y : array-like
        Data the mechanisms are run on; only its length is reported.
    epsilons : array-like
        Per-mechanism privacy parameters, strictly positive.
    deltas : array-like or None
        Per-mechanism delta parameters; None means all zero (pure DP).
    """
    eps = core.vec(epsilons)
    if len(eps) == 0:
        raise ValueError("k_step_dp_composition: epsilons is empty")
    for v in eps:
        if v <= 0:
            raise ValueError("k_step_dp_composition: epsilons must be positive")
    if deltas is None:
        dl = [0.0] * len(eps)
    else:
        dl = core.vec(deltas)
        if len(dl) != len(eps):
            raise ValueError("k_step_dp_composition: deltas and epsilons have different lengths")
        for v in dl:
            if v < 0 or v >= 1:
                raise ValueError("k_step_dp_composition: deltas must lie in [0, 1)")
    n = len(core.vec(y)) if y is not None else 0
    tot = sum(eps)
    return RichResult(
        title="Sequential composition of k eps-DP mechanisms",
        summary_lines=[("mechanisms", len(eps)), ("total epsilon", tot)],
        payload={
            "estimate": tot,
            "epsilon_total": tot,
            "delta_total": sum(dl),
            "k": float(len(eps)),
            "epsilon_max": max(eps),
            "epsilon_min": min(eps),
            "epsilon_mean": tot / len(eps),
            "laplace_scale": [1.0 / v for v in eps],
            "pure_dp": 1.0 if sum(dl) == 0.0 else 0.0,
            "n": n,
            "method": "eps_total = sum eps_i, delta_total = sum delta_i, Dwork & Roth (2014) Thm 3.16",
        },
    )


def cheatsheet():
    return "kcompo: Sequential composition of k epsilon-DP mechanisms"


# compact alias per ledger/NAMING.md
kstepdpcomposition = k_step_dp_composition
