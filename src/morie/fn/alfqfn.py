# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero action value Q(s, a) as the mean of the backed-up returns.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020),
arXiv:1911.08265, appendix B, which writes the action value as the mean
of the values backed up through the edge,

    Q(s,a) = ( sum over simulations that passed through (s,a) of v ) / N(s,a)

i.e. W(s,a) / N(s,a) in the AlphaGo Zero notation (Silver et al.,
*Nature* 550, 354-359).  With the visit counts N(s,a,z) of each distinct
backed-up return v(z), this is exactly the module's own formula line

    Q(s,a) = sum_z N(s,a,z) v(z) / N(s,a).

An unvisited edge has N = 0; AlphaGo Zero initialises its Q to zero,
which is what ``unvisited=0.0`` (the default) does here.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_q_function"]


def alphazero_q_function(N, v, unvisited=0.0):
    """Mean backed-up return per edge.

    Parameters
    ----------
    N : array-like
        Either the visit counts N(s,a) (1-D, paired elementwise with
        ``v`` giving the total backed-up value W(s,a)) or the counts
        N(s,a,z) per distinct return (2-D, rows = actions, cols = the
        returns listed in ``v``).
    v : array-like
        The backed-up value W(s,a) per action (1-D case) or the distinct
        returns v(z) (2-D case).
    unvisited : float
        Q assigned to an edge with N(s,a) = 0.

    Returns
    -------
    RichResult with payload:
        estimate : Q of the first action
        q        : Q(s,a) for every action
        w        : W(s,a) for every action
        n        : N(s,a) for every action
    """
    two_d = bool(N) and isinstance(N[0], (list, tuple))
    if two_d:
        rows = k.mat(N)
        vz = k.vec(v)
        q, w, nn = [], [], []
        for r in rows:
            tot = 0.0
            wt = 0.0
            for j in range(len(r)):
                tot += r[j]
                wt += r[j] * vz[j]
            nn.append(tot)
            w.append(wt)
            q.append(wt / tot if tot > 0.0 else float(unvisited))
    else:
        nn = k.vec(N)
        w = k.vec(v)
        q = [w[a] / nn[a] if nn[a] > 0.0 else float(unvisited) for a in range(len(nn))]
    return RichResult(
        title="AlphaZero action value",
        summary_lines=[("actions", len(q))],
        payload={
            "estimate": q[0] if q else float("nan"),
            "q": q,
            "w": w,
            "n": nn,
            "method": "AlphaZero action value Q(s,a) = W(s,a) / N(s,a)",
        },
    )


def cheatsheet():
    return "alfqfn: AlphaZero action-value Q via mean of child returns"
