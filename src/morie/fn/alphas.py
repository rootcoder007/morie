# morie.fn -- slice s03 (rootcoder007/morie)
"""One AlphaZero self-play game and the training targets it produces.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815.
A game is played from the current position by running MCTS at every
move and playing a_t ~ pi_t, where pi_t is the normalised root visit
count; the game record is stored as (s_t, pi_t, z), with z the final
outcome *from the point of view of the player to move at t*.  Those
triples are the regression targets for the loss
l = (z - v)^2 - pi' log p + c ||theta||^2.

DETERMINISM.  Move selection is greedy once the temperature has decayed
(AlphaGo Zero: after 30 moves) and, while the temperature is still one,
uses the inverse-CDF of pi at a van der Corput point rather than a
pseudo-random draw.  The rollout budget is a fixed number of
simulations.  No clock, no seed.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .alpz import alphazero_search

__all__ = ["alphazero_self_play"]


def alphazero_self_play(state, policy, value=None, mcts_iter=16, step=None,
                        terminal=None, outcome=None, max_moves=32,
                        temp_threshold=30, c_puct=1.25):
    """Play one self-play game and return its (s, pi, z) record.

    Parameters
    ----------
    state : scalar
        Starting state id.
    policy : callable
        ``s -> (p, v)``, or ``s -> p`` when ``value`` is given separately.
    value : callable, optional
        ``s -> v``, used when ``policy`` returns priors only.
    mcts_iter : int
        Simulations per move -- a fixed budget, never a time budget.
    step, terminal, outcome : callable, optional
        Environment: successor id, terminal test, and final outcome.
    max_moves : int
        Hard cap on the game length.
    temp_threshold : int
        Moves played at temperature one before selection turns greedy.

    Returns
    -------
    RichResult with payload:
        estimate : the game outcome z from the starting player's view
        states, pis, zs : the training record
        moves    : number of moves played
    """
    def net(s):
        out = policy(s)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return (out, value(s) if value is not None else 0.0)

    s = state
    states = []
    pis = []
    acts = []
    m = 0
    while m < int(max_moves):
        if terminal is not None and terminal(s):
            break
        res = alphazero_search(s, net, mcts_iter, step=step, c_puct=c_puct,
                               terminal=terminal)
        pi = res["pi"]
        states.append(s)
        pis.append(pi)
        if m >= int(temp_threshold):
            a = res["action"]
        else:
            u = k.vdc(m, 2)
            c = 0.0
            a = len(pi) - 1
            for j in range(len(pi)):
                c += pi[j]
                if u < c:
                    a = j
                    break
        acts.append(a)
        if step is None:
            break
        s = step(s, a)
        m += 1
    z = float(outcome(s)) if outcome is not None else 0.0
    zs = []
    sign = 1.0
    for _ in range(len(states)):
        zs.append(z * sign)
        sign = -sign
    return RichResult(
        title="AlphaZero self-play",
        summary_lines=[("moves", len(states)), ("outcome", z)],
        payload={
            "estimate": z,
            "states": states,
            "pis": pis,
            "actions": acts,
            "zs": zs,
            "moves": len(states),
            "final_state": s,
            "method": "AlphaZero self-play game producing (s, pi, z) targets",
        },
    )


def cheatsheet():
    return "alphas: AlphaZero self-play training step"
