# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero temperature decay for move selection.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815,
and Silver et al. (2017), *Nature* 550, 354-359.  During self-play a
move is sampled in proportion to the root visit counts raised to 1/tau,

    pi(a | s) = N(s,a)^(1/tau) / sum_b N(s,b)^(1/tau)

with tau = 1 for the first ``threshold`` moves of a game -- AlphaGo Zero
uses 30 -- and tau -> 0 thereafter, which makes the selection greedy.
Passing tau exactly to zero would be a division by zero, so the greedy
branch is taken directly: all mass on the most-visited action, ties
broken by the lowest action index.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_temp_decay"]


def alphazero_temp_decay(move_count, threshold=30, N=None):
    """Temperature for move ``move_count``, and the resulting policy.

    Parameters
    ----------
    move_count : int
        Zero-based index of the move about to be played.
    threshold : int
        Number of opening moves played at tau = 1.
    N : array-like, optional
        Root visit counts; when given, the sampling policy pi is returned.

    Returns
    -------
    RichResult with payload:
        estimate : tau
        tau      : same as estimate
        greedy   : whether the greedy branch applies
        pi       : the move-selection distribution (empty if N is None)
    """
    mc = int(move_count)
    th = int(threshold)
    greedy = mc >= th
    tau = 0.0 if greedy else 1.0
    pi = []
    if N is not None:
        n = k.vec(N)
        if greedy:
            best = 0
            for a in range(1, len(n)):
                if n[a] > n[best]:
                    best = a
            pi = [1.0 if a == best else 0.0 for a in range(len(n))]
        else:
            tot = 0.0
            for x in n:
                tot += x
            pi = [x / tot if tot > 0.0 else 0.0 for x in n]
    return RichResult(
        title="AlphaZero temperature decay",
        summary_lines=[("tau", tau), ("greedy", greedy)],
        payload={
            "estimate": tau,
            "tau": tau,
            "greedy": greedy,
            "pi": pi,
            "threshold": th,
            "method": "AlphaZero temperature schedule (tau = 1 then greedy)",
        },
    )


def cheatsheet():
    return "agtmpd: AlphaZero temperature decay schedule"
