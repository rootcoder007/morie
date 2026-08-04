# morie.fn -- slice s03 (rootcoder007/morie)
"""MCTS selection: the PUCT and UCT action rules.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020).  Mastering
Atari, Go, chess and shogi by planning with a learned model.
arXiv:1911.08265, appendix B, which prints the selection rule in full:

    a^k = argmax_a [ Q(s,a) + P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
                     * ( c1 + log( (sum_b N(s,b) + c2 + 1) / c2 ) ) ]

with c1 = 1.25 and c2 = 19652 in their experiments.  The AlphaGo Zero /
AlphaZero rule (Silver et al., *Nature* 550, 354-359, 2017; Silver et
al., arXiv:1712.01815 -- FETCHED, which states only that its search is
"identical to AlphaGo Zero") is the c2 -> infinity limit of the above,
because log((N + c2 + 1)/c2) -> 0, leaving the familiar

    U(s,a) = c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a)).

Both are provided; ``rule="puct"`` is the AlphaGo Zero limit and is the
default, ``rule="muzero"`` keeps the c2 term, ``rule="uct"`` is Kocsis
and Szepesvari's original

    UCT: argmax_a Q(s,a) + c sqrt(log sum_b N(s,b) / N(s,a)).

Deterministic throughout: ties are broken by the lowest action index,
never by a random draw.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["mcts_selection"]


def mcts_selection(Q, N, P, c=1.25, rule="puct", c2=19652.0):
    """Score every action and return the one PUCT/UCT would select.

    Parameters
    ----------
    Q : array-like
        Action values Q(s, a).
    N : array-like
        Visit counts N(s, a).
    P : array-like
        Prior probabilities P(s, a) from the policy network.
    c : float
        c_puct (or c1 for ``rule="muzero"``, or the UCT constant).
    rule : {"puct", "muzero", "uct"}
        Which selection rule to apply.
    c2 : float
        The MuZero c2 constant; ignored unless ``rule="muzero"``.

    Returns
    -------
    RichResult with payload:
        estimate  : index of the selected action (0-based)
        action    : same as estimate
        scores    : the score of every action
        u         : the exploration bonus of every action
        n_total   : sum_b N(s, b)
    """
    q = k.vec(Q)
    n = k.vec(N)
    p = k.vec(P)
    m = len(q)
    tot = 0.0
    for v in n:
        tot += v
    root = math.sqrt(tot) if tot > 0.0 else 0.0
    u = [0.0] * m
    scores = [0.0] * m
    for a in range(m):
        if rule == "uct":
            if n[a] > 0.0 and tot > 0.0:
                u[a] = c * math.sqrt(math.log(tot) / n[a])
            else:
                u[a] = float("inf")
        else:
            base = p[a] * root / (1.0 + n[a])
            if rule == "muzero":
                base = base * (c + math.log((tot + c2 + 1.0) / c2))
            else:
                base = base * c
            u[a] = base
        scores[a] = q[a] + u[a]
    best = 0
    for a in range(1, m):
        if scores[a] > scores[best]:
            best = a
    return RichResult(
        title="MCTS selection",
        summary_lines=[("selected action", best), ("rule", rule)],
        payload={
            "estimate": float(best),
            "action": best,
            "scores": scores,
            "u": u,
            "n_total": tot,
            "rule": rule,
            "method": "MCTS selection by " + str(rule).upper(),
        },
    )


def cheatsheet():
    return "mctsel: MCTS selection phase via UCT or PUCT"


mctsselection = mcts_selection
