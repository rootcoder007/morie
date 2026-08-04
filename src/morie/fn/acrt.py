# morie.fn -- slice s03 (rootcoder007/morie)
"""One-step actor-critic.

Source consulted (FETCHED from incompleteideas.net): Sutton, R. S. and
Barto, A. G. (2018).  *Reinforcement Learning: An Introduction*, 2nd
edition, section 13.5, equations (13.12)-(13.14):

    theta_(t+1) = theta_t + alpha ( G_(t:t+1) - vhat(S_t, w) )
                            grad pi(A_t | S_t, theta_t) / pi(A_t | S_t, theta_t)
                = theta_t + alpha ( R_(t+1) + gamma vhat(S_(t+1), w)
                                    - vhat(S_t, w) )
                            grad ln pi(A_t | S_t, theta_t)
                = theta_t + alpha delta_t grad ln pi(A_t | S_t, theta_t)

paired with semi-gradient TD(0) for the critic,

    w_(t+1) = w_t + alpha_w delta_t grad vhat(S_t, w).

The episodic form in the book's pseudocode carries the discount factor I
(I <- gamma I after each step), which is included here as
``discount_actor``.  The gradients are supplied by the caller, since
they belong to whatever parameterisation the policy uses.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["actor_critic"]


def actor_critic(env, actor=None, critic=None, rewards=None, values=None,
                 grad_logpi=None, grad_v=None, alpha_theta=0.1, alpha_w=0.1,
                 gamma=0.99, theta=None, w=None, discount_actor=True):
    """Run the one-step actor-critic updates over a trajectory.

    Parameters
    ----------
    env : array-like
        The rewards R_1..R_T.  (Positional, for signature stability.)
    actor, critic : array-like, optional
        Alternative slots for ``grad_logpi`` and ``values``.
    rewards, values : array-like, optional
        Rewards, and vhat(S_0..S_T) -- one more value than rewards.
    grad_logpi : 2-D array-like, optional
        grad ln pi(A_t | S_t) per step, one row per step.
    grad_v : 2-D array-like, optional
        grad vhat(S_t) per step; defaults to a single unit feature.
    alpha_theta, alpha_w : float
        Actor and critic step sizes.
    gamma : float
        Discount.
    theta, w : array-like, optional
        Starting parameters; zeros by default.
    discount_actor : bool
        Carry the book's I factor, I <- gamma I.

    Returns
    -------
    RichResult with payload:
        estimate : the mean TD error over the trajectory
        deltas   : delta_t per step
        theta, w : the updated parameters
    """
    R = k.vec(rewards if rewards is not None else env)
    V = k.vec(values if values is not None else (critic if critic is not None else []))
    T = len(R)
    G = k.mat(grad_logpi if grad_logpi is not None else actor) if (
        grad_logpi is not None or actor is not None) else [[1.0]] * T
    Gv = k.mat(grad_v) if grad_v is not None else [[1.0]] * T
    th = k.vec(theta) if theta is not None else [0.0] * (len(G[0]) if G else 1)
    ww = k.vec(w) if w is not None else [0.0] * (len(Gv[0]) if Gv else 1)
    g = float(gamma)
    deltas = []
    I = 1.0
    for t in range(T):
        vt = V[t] if t < len(V) else 0.0
        vn = V[t + 1] if t + 1 < len(V) else 0.0
        d = R[t] + g * vn - vt
        deltas.append(d)
        for j in range(len(th)):
            th[j] = th[j] + float(alpha_theta) * I * d * G[t][j]
        for j in range(len(ww)):
            ww[j] = ww[j] + float(alpha_w) * d * Gv[t][j]
        if discount_actor:
            I = I * g
    return RichResult(
        title="One-step actor-critic",
        summary_lines=[("steps", T)],
        payload={
            "estimate": k.mean(deltas) if deltas else float("nan"),
            "deltas": deltas,
            "theta": th,
            "w": ww,
            "n": T,
            "method": "One-step actor-critic (Sutton and Barto 2018, eqs. 13.12-13.14)",
        },
    )


def cheatsheet():
    return "acrt: Actor-critic with TD baseline"
