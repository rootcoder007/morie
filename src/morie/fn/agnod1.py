# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero MCTS node initialisation.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020),
arXiv:1911.08265, appendix B, and Silver et al. (2017), *Nature* 550,
354-359.  A newly created node stores, for each edge (s,a),

    {N(s,a) = 0, W(s,a) = 0, Q(s,a) = 0, P(s,a) = p_a}

The only content of the node beyond the priors is zero -- which is the
point: AlphaZero carries no rollout statistics and no heuristic, so a
node is fully described by the policy prior until the first backup
reaches it.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_node_init"]


def alphazero_node_init(p, action_space=None):
    """Create the edge statistics of a fresh MCTS node.

    Parameters
    ----------
    p : array-like
        Prior probabilities from the policy head.
    action_space : int, optional
        Size of the action space.  When given and larger than ``p``, the
        priors are zero-padded; when smaller, ``p`` is truncated.

    Returns
    -------
    RichResult with payload:
        estimate  : the number of edges created
        p         : the (padded/truncated, renormalised) priors
        n, w, q   : zero vectors of that length
        prior_sum : sum of the stored priors, 1 when p was a distribution
    """
    pr = k.vec(p)
    if action_space is not None:
        m = int(action_space)
        pr = pr[:m] + [0.0] * (m - len(pr)) if len(pr) < m else pr[:m]
    m = len(pr)
    tot = 0.0
    for x in pr:
        tot += x
    if tot > 0.0:
        pr = [x / tot for x in pr]
        tot = 1.0
    return RichResult(
        title="AlphaZero MCTS node",
        summary_lines=[("edges", m)],
        payload={
            "estimate": float(m),
            "p": pr,
            "n": [0.0] * m,
            "w": [0.0] * m,
            "q": [0.0] * m,
            "prior_sum": tot,
            "method": "AlphaZero MCTS node initialisation (N=W=Q=0, P=p)",
        },
    )


def cheatsheet():
    return "agnod1: AlphaZero MCTS node initialization"
