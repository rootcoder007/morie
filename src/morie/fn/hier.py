# morie.fn -- slice s03 (rootcoder007/morie)
"""Hierarchical RL: the options framework.

Source consulted: Sutton, R. S., Precup, D. and Singh, S. (1999).
Between MDPs and semi-MDPs: a framework for temporal abstraction in
reinforcement learning.  *Artificial Intelligence* 112(1-2), 181-211.
An option is the triple

    omega = (I, pi_omega, beta_omega)

-- an initiation set, an internal policy, and a termination condition --
and, because an option occupies k time steps, the value backup is the
SMDP one:

    Q(s, omega) <- Q(s, omega)
                   + alpha [ r + gamma^k max_omega' Q(s', omega')
                             - Q(s, omega) ]

with r the discounted reward accumulated over those k steps,
r = R_1 + gamma R_2 + ... + gamma^(k-1) R_k.  The gamma^k, rather than a
plain gamma, is the whole content of the semi-Markov generalisation.
The 1999 AIJ paper is paywalled; the definition of an option and the
SMDP backup are quoted in their standard published form, which is
reproduced identically in Sutton and Barto (2018) section 17.2 (FETCHED
from incompleteideas.net).

DETERMINISM.  Option termination uses the caller's beta as a threshold
against van der Corput points rather than a random draw.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["hierarchical_rl"]


def hierarchical_rl(env, options=None, meta=None, rewards=None, gamma=0.99,
                    alpha=0.1, Q=None, q_next=None, k_steps=None):
    """SMDP option-value backup for one executed option.

    Parameters
    ----------
    env : array-like
        The rewards R_1..R_k received while the option ran.
    options : array-like, optional
        The termination probabilities beta_omega per step; the option is
        taken to end at the first step where beta exceeds the van der
        Corput point, or after all rewards when none does.
    meta : float, optional
        The current Q(s, omega); zero by default.
    rewards : array-like, optional
        Explicit rewards, overriding ``env``.
    gamma : float
        Discount.
    alpha : float
        Step size.
    Q : float, optional
        Alternative slot for the current Q(s, omega).
    q_next : array-like, optional
        Q(s', omega') for every option at the arrival state.
    k_steps : int, optional
        Force the option duration instead of deriving it from beta.

    Returns
    -------
    RichResult with payload:
        estimate  : the updated Q(s, omega)
        r_option  : the discounted reward accumulated over the option
        k         : the option duration
        target    : r + gamma^k max Q(s', .)
        td_error  : target - Q(s, omega)
    """
    R = k.vec(rewards if rewards is not None else env)
    g = float(gamma)
    if k_steps is not None:
        kk = int(k_steps)
    elif options is not None:
        beta = k.vec(options)
        kk = len(R)
        for t in range(len(beta)):
            if k.vdc(t, 2) < beta[t]:
                kk = t + 1
                break
    else:
        kk = len(R)
    if kk > len(R):
        kk = len(R)
    r = 0.0
    for t in range(kk):
        r += (g ** t) * R[t]
    q0 = float(Q if Q is not None else (meta if meta is not None else 0.0))
    qn = k.vec(q_next) if q_next is not None else []
    mx = max(qn) if qn else 0.0
    target = r + (g ** kk) * mx
    td = target - q0
    return RichResult(
        title="Options framework (SMDP backup)",
        summary_lines=[("k", kk), ("Q", q0 + float(alpha) * td)],
        payload={
            "estimate": q0 + float(alpha) * td,
            "r_option": r,
            "k": kk,
            "target": target,
            "td_error": td,
            "method": "SMDP option-value backup (Sutton, Precup and Singh 1999)",
        },
    )


def cheatsheet():
    return "hier: Hierarchical RL (options framework)"


hierarchicalrl = hierarchical_rl
