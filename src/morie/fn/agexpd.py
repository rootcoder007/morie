# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero MCTS expansion of a newly reached state.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020),
arXiv:1911.08265, appendix B ("Expansion"), and Silver et al. (2017),
*Nature* 550, 354-359.  When a simulation reaches a state that is not
yet in the tree, the network is evaluated once,

    (p, v) = f_theta(s)

the prior of each edge is set to P(s,a) = p_a, and the edge statistics
are initialised to N(s,a) = W(s,a) = Q(s,a) = 0.  Illegal moves are
masked out and the remaining priors renormalised to sum to one -- the
network's policy head is over the whole action space, so the mask is
what makes p a distribution over the *legal* moves.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_expand"]


def alphazero_expand(state, policy_net, legal=None, logits=False):
    """Evaluate the policy network at a leaf and initialise its edges.

    Parameters
    ----------
    state : any
        The state being expanded; carried through untouched.
    policy_net : callable or array-like
        Either a callable ``s -> (p, v)`` / ``s -> p``, or the prior
        vector itself.
    legal : array-like, optional
        Legal-move mask (1/0 or booleans) over the action space.
    logits : bool
        Treat the network output as logits and apply a softmax first.

    Returns
    -------
    RichResult with payload:
        estimate : the value v (nan when the net returns priors only)
        p        : masked, renormalised priors
        n, w, q  : freshly initialised edge statistics (all zero)
    """
    out = policy_net(state) if callable(policy_net) else policy_net
    v = float("nan")
    if isinstance(out, tuple) and len(out) == 2:
        raw, v = k.vec(out[0]), float(out[1])
    else:
        raw = k.vec(out)
    if logits:
        raw = k.softmax(raw)
    m = len(raw)
    mask = [1.0] * m if legal is None else [1.0 if x else 0.0 for x in legal]
    masked = [raw[a] * mask[a] for a in range(m)]
    tot = 0.0
    for x in masked:
        tot += x
    if tot > 0.0:
        p = [x / tot for x in masked]
    else:
        live = 0.0
        for x in mask:
            live += x
        p = [x / live if live > 0.0 else 0.0 for x in mask]
    return RichResult(
        title="AlphaZero MCTS expansion",
        summary_lines=[("actions", m), ("value", v)],
        payload={
            "estimate": v,
            "value": v,
            "p": p,
            "n": [0.0] * m,
            "w": [0.0] * m,
            "q": [0.0] * m,
            "state": state,
            "method": "AlphaZero MCTS expansion via the policy network",
        },
    )


def cheatsheet():
    return "agexpd: AlphaZero MCTS expansion via policy network"
