# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaGo's fast rollout policy and its mixed leaf evaluation.

Source consulted: Silver, D. et al. (2016).  Mastering the game of Go
with deep neural networks and tree search.  *Nature* 529, 484-489.  The
Nature paper is paywalled, but the two equations it turns on are quoted
identically in every account of the system and in its successor papers
(Silver et al. 2017, *Nature* 550, 354-359; Silver et al. 2018,
arXiv:1712.01815 -- FETCHED, which describes AlphaGo Zero as removing
exactly this rollout):

    V(s_L) = (1 - lambda) v_theta(s_L) + lambda z_L

the leaf value is a convex mixture of the value network v_theta and the
outcome z_L of a fast rollout played to the end of the game with the
linear-softmax rollout policy pi_rollout; AlphaGo used lambda = 0.5.

DETERMINISM.  The rollout is not sampled from a generator.  Actions come
either from a caller-supplied action stream, or -- the default -- from
the inverse CDF of pi_rollout evaluated at van der Corput points, which
gives a reproducible spread of moves without a seed.  The horizon is a
fixed number of plies.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphago_montecarlo"]


def alphago_montecarlo(state, rollout_net, horizon=16, step=None,
                       terminal=None, outcome=None, value_net=None,
                       lam=0.5, stream=None):
    """Play a fast rollout and mix its outcome with the value network.

    Parameters
    ----------
    state : scalar
        Leaf state id.
    rollout_net : callable
        ``s -> p``, the fast rollout policy over actions.
    horizon : int
        Maximum number of plies rolled out.
    step, terminal, outcome : callable, optional
        Environment.
    value_net : callable, optional
        ``s -> v``; when absent lambda is forced to 1 (pure rollout).
    lam : float
        Mixing weight; AlphaGo used 0.5.
    stream : array-like, optional
        Uniforms in [0, 1) driving the rollout, one per ply.

    Returns
    -------
    RichResult with payload:
        estimate  : V(s_L), the mixed leaf value
        z         : the rollout outcome
        v_theta   : the value-network estimate (nan when absent)
        plies     : number of plies actually played
        trajectory: the states visited
    """
    s = state
    traj = [s]
    acts = []
    i = 0
    while i < int(horizon):
        if terminal is not None and terminal(s):
            break
        p = k.vec(rollout_net(s))
        tot = 0.0
        for x in p:
            tot += x
        if tot > 0.0:
            p = [x / tot for x in p]
        u = float(stream[i]) if stream is not None and i < len(stream) else k.vdc(i, 2)
        c = 0.0
        a = len(p) - 1
        for j in range(len(p)):
            c += p[j]
            if u < c:
                a = j
                break
        acts.append(a)
        if step is None:
            break
        s = step(s, a)
        traj.append(s)
        i += 1
    z = float(outcome(s)) if outcome is not None else 0.0
    if value_net is None:
        vt = float("nan")
        val = z
    else:
        vt = float(value_net(state))
        L = float(lam)
        val = (1.0 - L) * vt + L * z
    return RichResult(
        title="AlphaGo rollout evaluation",
        summary_lines=[("V(s_L)", val), ("plies", i)],
        payload={
            "estimate": val,
            "z": z,
            "v_theta": vt,
            "lam": float(lam),
            "plies": i,
            "trajectory": traj,
            "actions": acts,
            "method": "AlphaGo mixed leaf value (1-lambda) v_theta + lambda z_L",
        },
    )


def cheatsheet():
    return "alfgom: AlphaGo Monte Carlo rollout policy"
