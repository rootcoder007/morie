# morie.fn -- function file (rootcoder007/morie)
"""Deep deterministic policy gradient (continuous actions)."""

from __future__ import annotations

from .hmddpg import geron_ddpg

__all__ = ["ddpg"]


def ddpg(env, actor, critic, tau=0.01, epochs=20, lr=0.01, gamma=0.95, ou_theta=0.15, ou_sigma=0.2, seed=0, s0=None):
    """Deep deterministic policy gradient (continuous actions).

    Off-policy actor-critic for continuous actions (Lillicrap et al. 2016):
    deterministic actor ``mu(s)``, critic ``Q(s, a)`` trained on the TD error
    against target networks, actor moved along the deterministic policy
    gradient ``dQ/da dmu/dtheta`` (Silver et al. 2014), targets tracked with
    the soft update ``theta' <- tau theta + (1 - tau) theta'`` and
    Ornstein-Uhlenbeck exploration. This is :func:`morie.fn.hmddpg.geron_ddpg`
    (linear actor and critic).

    References
    ----------
    Lillicrap, T. P. et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.

    Silver, D. et al. (2014). Deterministic policy gradient algorithms. *ICML*, 387-395.

    Examples
    --------
    >>> import inspect
    >>> "tau" in inspect.signature(ddpg).parameters
    True
    """
    return geron_ddpg(
        env,
        actor,
        critic,
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        tau=tau,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        seed=seed,
        s0=s0,
    )


def cheatsheet():
    return "ddpgc: DDPG (front-end to hmddpg.geron_ddpg)"
