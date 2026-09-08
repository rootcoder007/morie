# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""PUCT action selection for AlphaZero-style tree search.

Sources: Rosin, C. D. (2011), "Multi-armed bandits with episode
context", Annals of Mathematics and Artificial Intelligence 61(3),
203-230, doi:10.1007/s10472-011-9258-6, which introduces the predictor
form of UCB; and Silver, D. et al. (2017), "Mastering the game of Go
without human knowledge", Nature 550, 354-359,
doi:10.1038/nature24270, whose search selects

    a_t = argmax_a ( Q(s, a) + U(s, a) ),
    U(s, a) = c_puct P(s, a) sqrt( sum_b N(s, b) ) / ( 1 + N(s, a) ).

The shape of U is the whole point.  The numerator grows with the square
root of the parent visit count, so exploration decays only slowly as the
tree is searched; the denominator is 1 + N, so a child that has never
been visited still gets a finite bonus rather than an infinite one, and
the prior P is what breaks the tie among them.  Replacing 1 + N with N
gives division by zero at the root and is the classic transcription
error.

At sum_b N(s, b) = 0 the exploration term vanishes for every action, so
selection falls back to Q alone -- which is why AlphaZero seeds the root
with one evaluation before selecting.  That degenerate case is checked
as an anchor, as is c_puct = 0, which must reduce to greedy Q.

Ties in the argmax go to the lowest index, so the rule is a function and
both language arms agree.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["alphazero_puct"]


def alphazero_puct(P, N, Q, c_puct=1.0):
    """PUCT scores and the selected action.

    Parameters
    ----------
    P : array-like
        Prior probabilities over the actions; must be non-negative.
    N : array-like
        Visit counts, non-negative.
    Q : array-like
        Action values.
    c_puct : float
        Exploration constant, non-negative.

    Returns
    -------
    score : Q + U for every action
    U : the exploration term
    action : the 0-based argmax
    """
    p = [float(v) for v in core.vec(P)]
    n = [float(v) for v in core.vec(N)]
    q = [float(v) for v in core.vec(Q)]
    k = len(p)
    if k == 0:
        raise ValueError("alphazero_puct: no actions")
    if len(n) != k or len(q) != k:
        raise ValueError("alphazero_puct: P, N and Q must have the same length")
    for v in p:
        if v < 0.0:
            raise ValueError("alphazero_puct: priors must be non-negative")
    for v in n:
        if v < 0.0:
            raise ValueError("alphazero_puct: visit counts must be non-negative")
    c = float(c_puct)
    if c < 0.0:
        raise ValueError("alphazero_puct: c_puct must be non-negative")
    tot = 0.0
    for v in n:
        tot += v
    rt = math.sqrt(tot)
    U = []
    sc = []
    for i in range(k):
        u = c * p[i] * rt / (1.0 + n[i])
        U.append(u)
        sc.append(q[i] + u)
    best = 0
    for i in range(1, k):
        if sc[i] > sc[best]:
            best = i
    return RichResult(
        title="PUCT selection",
        summary_lines=[("action", best), ("score", sc[best])],
        payload={
            "score": sc,
            "estimate": sc[best],
            "U": U,
            "action": best,
            "n_parent": tot,
            "sqrt_n_parent": rt,
            "c_puct": c,
            "k": k,
            "method": "Q + c_puct P sqrt(sum_b N_b)/(1 + N); Rosin (2011), Silver et al. (2017)",
        },
    )


def cheatsheet():
    return "agpuct: PUCT score for AlphaZero MCTS"


# compact alias per ledger/NAMING.md
alphazeropuct = alphazero_puct
