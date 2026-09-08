# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero MCTS backup along the simulation path.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020),
arXiv:1911.08265, appendix B ("Backup"), and Silver et al. (2017),
*Nature* 550, 354-359, whose backup is the same in the two-player,
undiscounted case.  For every edge (s,a) on the path from the root to
the expanded leaf,

    N(s,a) <- N(s,a) + 1
    W(s,a) <- W(s,a) + G
    Q(s,a) <- W(s,a) / N(s,a)

where G is the value returned by the network at the leaf.  In a
two-player zero-sum game the value is negated at each ply as it is
propagated back up, because a value is always expressed from the point
of view of the player to move -- this is the ``alternate`` flag, on by
default, and it is what makes the backup a *minimax* backup rather than
a plain average.  MuZero's general form adds discounting and the
intermediate rewards r_k, which are accepted here as ``rewards``.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_backup"]


def alphazero_backup(leaf, value, path, N=None, W=None, rewards=None,
                     gamma=1.0, alternate=True):
    """Propagate a leaf value back along the search path.

    Parameters
    ----------
    leaf : any
        The expanded leaf; carried through to the result untouched.
    value : float
        The value the network assigned to ``leaf``.
    path : array-like
        The edges from the root to the leaf, in root-first order.  Only
        its length is used to index ``N`` and ``W``.
    N, W : array-like, optional
        Current visit counts and total values for the edges on the path,
        root-first.  Default: all zero.
    rewards : array-like, optional
        MuZero's intermediate rewards r_k per edge, root-first.
    gamma : float
        Discount applied per ply.
    alternate : bool
        Negate the value at each ply (two-player zero-sum).

    Returns
    -------
    RichResult with payload:
        estimate : the value backed up into the root edge
        n, w, q  : updated statistics per edge, root-first
        g        : the return credited to each edge, root-first
    """
    L = len(list(path))
    n = k.vec(N) if N is not None else [0.0] * L
    w = k.vec(W) if W is not None else [0.0] * L
    r = k.vec(rewards) if rewards is not None else [0.0] * L
    g = [0.0] * L
    acc = float(value)
    # walk leaf-ward edge first, exactly as the backup runs
    for i in range(L - 1, -1, -1):
        if alternate:
            acc = -acc
        acc = r[i] + float(gamma) * acc
        g[i] = acc
        n[i] = n[i] + 1.0
        w[i] = w[i] + acc
    q = [w[i] / n[i] if n[i] > 0.0 else 0.0 for i in range(L)]
    return RichResult(
        title="AlphaZero MCTS backup",
        summary_lines=[("edges updated", L)],
        payload={
            "estimate": q[0] if q else float("nan"),
            "n": n,
            "w": w,
            "q": q,
            "g": g,
            "leaf": leaf,
            "method": "AlphaZero MCTS backup along the simulation path",
        },
    )


def cheatsheet():
    return "agbckp: AlphaZero MCTS backup"
