# morie.fn -- function file (rootcoder007/morie)
"""Twin-delayed DDPG (TD3)."""

from .hmtd3 import geron_td3


def td3(env, actor=None, critic1=None, critic2=None, **kwargs):
    r"""Twin-delayed deep deterministic policy gradient (TD3; Fujimoto, van Hoof and Meger 2018).

    Twin critics regressed on ``r + gamma min(Q1', Q2')(s', a~)``, target
    policy smoothing and delayed policy / target updates, over a finite
    action set. ``actor`` is the initial deterministic policy (one action per
    state) and ``critic1``/``critic2`` the initial critics; other arguments
    (``epochs``, ``lr``, ``gamma``, ``steps``, ``policy_delay``, ``tau``,
    ``noise``, ``seed``) pass through. Thin front-end to
    :func:`morie.fn.hmtd3.geron_td3`.

    References
    ----------
    Fujimoto, S., van Hoof, H. and Meger, D. (2018). Addressing function
    approximation error in actor-critic methods. *ICML*, PMLR 80,
    1587-1596.

    Examples
    --------
    >>> class Bandit:
    ...     n_states, n_actions = 1, 2
    ...     def reset(self):
    ...         return 0
    ...     def step(self, a):
    ...         return 0, float(a), False
    >>> int(td3(Bandit(), epochs=40)["policy"][0])
    1
    """
    return geron_td3(env, policy=actor, Q1=critic1, Q2=critic2, **kwargs)


def cheatsheet():
    return "td3c: TD3 (twin critics, target smoothing, delayed policy) via hmtd3.geron_td3"
