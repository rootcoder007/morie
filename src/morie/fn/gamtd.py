# morie.fn -- slice s03 (rootcoder007/morie)
"""The n-step TD return and its update.

Source consulted (FETCHED from incompleteideas.net): Sutton, R. S. and
Barto, A. G. (2018).  *Reinforcement Learning: An Introduction*, 2nd
edition, equation (7.1):

    G_(t:t+n) = R_(t+1) + gamma R_(t+2) + ... + gamma^(n-1) R_(t+n)
                + gamma^n V_(t+n-1)(S_(t+n))

"for all n, t such that n >= 1 and 0 <= t < T - n", with the convention
G_(t:t+n) = G_t whenever t + n >= T -- that is, the bootstrap term is
simply dropped once the return runs past the end of the episode.  The
n-step TD update, equation (7.2), is

    V_(t+n)(S_t) = V_(t+n-1)(S_t) + alpha [ G_(t:t+n) - V_(t+n-1)(S_t) ].

Both are computed here.  n = 1 recovers TD(0) and n >= T recovers Monte
Carlo, which is the point of the family.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["n_step_td"]


def n_step_td(traj, V, n=1, gamma=0.99, alpha=0.1, states=None):
    """n-step returns for a trajectory, and the resulting value updates.

    Parameters
    ----------
    traj : array-like
        The rewards R_1..R_T of the episode.
    V : array-like
        Current value estimates, indexed by the state index in ``states``
        (or by time step when ``states`` is None).
    n : int
        The number of steps before bootstrapping.
    gamma : float
        Discount.
    alpha : float
        Step size for the update.
    states : array-like, optional
        Zero-based state index visited at each time 0..T.

    Returns
    -------
    RichResult with payload:
        estimate : G_(0:n), the return from the first state
        returns  : G_(t:t+n) for every t
        v_new    : the updated value estimates
        bootstrapped : whether each return used the bootstrap term
    """
    R = k.vec(traj)
    v = k.vec(V)
    T = len(R)
    nn = int(n)
    g = float(gamma)
    idx = [int(s) for s in states] if states is not None else list(range(T + 1))
    G = []
    booted = []
    for t in range(T):
        h = t + nn
        acc = 0.0
        j = t
        while j < h and j < T:
            acc += (g ** (j - t)) * R[j]
            j += 1
        if h < T and h < len(idx):
            si = idx[h]
            acc += (g ** nn) * (v[si] if si < len(v) else 0.0)
            booted.append(1.0)
        else:
            booted.append(0.0)
        G.append(acc)
    vn = list(v)
    for t in range(T):
        si = idx[t] if t < len(idx) else t
        if si < len(vn):
            vn[si] = vn[si] + float(alpha) * (G[t] - vn[si])
    return RichResult(
        title="n-step TD",
        summary_lines=[("n", nn), ("steps", T)],
        payload={
            "estimate": G[0] if G else float("nan"),
            "returns": G,
            "v_new": vn,
            "bootstrapped": booted,
            "n": T,
            "method": "n-step TD return and update (Sutton and Barto 2018, eqs. 7.1-7.2)",
        },
    )


def cheatsheet():
    return "gamtd: n-step TD return"


nsteptd = n_step_td
