# morie.fn -- slice s03 (rootcoder007/morie)
"""Search-horizon control: truncate the search and bootstrap the value.

Source consulted (FETCHED): Schrittwieser, J. et al. (2020),
arXiv:1911.08265, appendix B, whose backup for a search truncated at
depth l is written out as

    G^k = sum_{tau = 0}^{l - 1 - k} gamma^tau r_(k + 1 + tau)
          + gamma^(l - k) v^l

that is, the discounted rewards accumulated down to the truncation depth
plus the discounted network value at the truncated leaf.  Silver et al.
(2018), arXiv:1712.01815 (FETCHED), uses the undiscounted two-player
special case, gamma = 1 and r = 0 except at the terminal node.

This is exactly the n-step return of Sutton and Barto (2018),
*Reinforcement Learning: An Introduction*, 2nd edition, equation (7.1)
(FETCHED from incompleteideas.net), with the bootstrap supplied by the
value network instead of a tabular estimate -- which is the point: the
horizon buys depth at the price of trusting v.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_search_horizon"]


def alphazero_search_horizon(depth_limit, state, rewards=None, values=None,
                             gamma=1.0, k_start=0):
    """Truncated return at a search horizon.

    Parameters
    ----------
    depth_limit : int
        The truncation depth l.
    state : any
        Carried through untouched; the root the return is credited to.
    rewards : array-like, optional
        Rewards r_1, r_2, ... along the principal variation.
    values : array-like, optional
        Value estimates v^d at each depth d; v^l is the bootstrap.
    gamma : float
        Discount.
    k_start : int
        The depth k the return is computed from.

    Returns
    -------
    RichResult with payload:
        estimate  : G^k
        bootstrap : the discounted bootstrap term
        reward_part : the discounted reward sum
        depth     : the effective truncation depth used
    """
    r = k.vec(rewards) if rewards is not None else []
    v = k.vec(values) if values is not None else []
    l = int(depth_limit)
    if l > len(r):
        l = len(r) if r else l
    kk = int(k_start)
    g = float(gamma)
    part = 0.0
    tau = 0
    while kk + tau < l:
        part += (g ** tau) * r[kk + tau]
        tau += 1
    if v:
        idx = l if l < len(v) else len(v) - 1
        boot = (g ** (l - kk)) * v[idx]
    else:
        boot = 0.0
    return RichResult(
        title="AlphaZero search horizon",
        summary_lines=[("G", part + boot), ("depth", l)],
        payload={
            "estimate": part + boot,
            "bootstrap": boot,
            "reward_part": part,
            "depth": l,
            "state": state,
            "method": "Truncated search return: discounted rewards + bootstrapped value",
        },
    )


def cheatsheet():
    return "agscho: AlphaZero search horizon control"
