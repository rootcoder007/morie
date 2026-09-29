# morie.fn -- function file (rootcoder007/morie)
"""Advantage actor-critic (synchronous A2C)."""

from __future__ import annotations

from .hma2c import geron_a2c

__all__ = ["a2c"]


def a2c(env, actor, critic, n_steps=200, epochs=100, lr=0.1, gamma=0.99, critic_lr=None, seed=0):
    """Advantage actor-critic (synchronous A2C).

    Linear-softmax policy and linear value baseline trained with the
    advantage ``A_t = G_t - V(s_t)`` (Mnih et al. 2016): ``actor += lr A_t
    (onehot(a_t) - pi(.|s_t)) s_t'``, ``critic += critic_lr A_t s_t`` over
    rollouts of at most ``n_steps`` steps. This is
    :func:`morie.fn.hma2c.geron_a2c` (``n_steps`` is its ``max_steps``);
    ``env`` offers ``reset() -> state`` and ``step(action) -> (state,
    reward, done)``.

    References
    ----------
    Mnih, V. et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*, 1928-1937.

    Geron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, 3rd ed.
    O'Reilly, ch. 18.

    Examples
    --------
    >>> env = {"reset": lambda: [1.0], "step": lambda a: ([1.0], 1.0 if a == 0 else 0.0, True)}
    >>> r = a2c(env, [[0.0], [0.0]], [0.0], n_steps=5, epochs=200, lr=0.5, seed=1)
    >>> bool(r["policy"]([1.0])[0] > 0.9)
    True
    """
    return geron_a2c(
        env, actor, critic, epochs=epochs, lr=lr, gamma=gamma, critic_lr=critic_lr, max_steps=n_steps, seed=seed
    )


def cheatsheet():
    return "a2cv: advantage actor-critic (front-end to hma2c.geron_a2c)"
