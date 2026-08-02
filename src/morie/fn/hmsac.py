# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Soft actor-critic (SAC): entropy-regularized max reward."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_sac"]


def _check_env(env):
    for attr in ("reset", "step"):
        if not callable(getattr(env, attr, None)):
            raise ValueError(f"geron_sac: env must provide a callable {attr}(); got {type(env).__name__}")
    n_s = getattr(env, "n_states", None)
    n_a = getattr(env, "n_actions", None)
    if not isinstance(n_s, (int, np.integer)) or not isinstance(n_a, (int, np.integer)):
        raise ValueError("geron_sac: env must expose integer n_states and n_actions for the tabular soft update")
    if n_s < 1 or n_a < 2:
        raise ValueError(f"geron_sac: need n_states >= 1 and n_actions >= 2, got {n_s} and {n_a}")
    return int(n_s), int(n_a)


def geron_sac(env, policy=None, critic=None, epochs=20, lr=0.5, alpha=0.2, gamma=0.9, steps=20, seed=0):
    """
    Soft actor-critic (SAC): entropy-regularized max reward.

    Formula: pi* = argmax E[sum_t r_t + alpha*H(pi(.|s_t))]

    The tabular soft actor-critic. Every piece of the algorithm is here
    and is exact rather than approximated by a sampled gradient:

    * soft state value ``V(s) = sum_a pi(a|s) (Q(s,a) - alpha log pi(a|s))``
      -- the entropy bonus enters the *value*, not just the objective;
    * soft critic update ``Q(s,a) <- Q + lr*(r + gamma V(s') - Q)``;
    * policy improvement in closed form,
      ``pi(.|s) = softmax(Q(s,.)/alpha)``, which is the exact maximiser
      of ``E_pi[Q] + alpha H(pi)`` (delegated to
      :func:`morie.fn.hmsftm.geron_softmax_function`).

    As ``alpha -> 0`` the policy becomes greedy and this reduces to
    Q-learning; as ``alpha -> inf`` it becomes uniform. Both limits are
    visible in the returned entropy trace.

    Parameters
    ----------
    env : object
        Must provide ``reset() -> s``, ``step(a) -> (s', r, done)`` and
        integer attributes ``n_states``, ``n_actions``.
    policy : array-like, optional
        Initial (n_states, n_actions) action probabilities; default uniform.
    critic : array-like, optional
        Initial (n_states, n_actions) Q table; default zeros.
    epochs : int, default 20
        Alternations of (collect, critic update, policy improvement).
    lr : float, default 0.5
        Critic step size in (0, 1].
    alpha : float, default 0.2
        Entropy temperature (> 0).
    gamma : float, default 0.9
        Discount in [0, 1).
    steps : int, default 20
        Environment steps collected per epoch.
    seed : int, default 0
        LCG seed for action sampling (no global RNG state is touched).

    Returns
    -------
    result : RichResult
        Keys: policy, Q, V, entropy, returns, estimate, n, method.

    Examples
    --------
    A one-state bandit where action 1 pays 1 and action 0 pays nothing.
    With a cold temperature the soft policy collapses onto the paying arm:

    >>> import numpy as np
    >>> class Bandit:
    ...     n_states, n_actions = 1, 2
    ...     def reset(self):
    ...         return 0
    ...     def step(self, a):
    ...         return 0, float(a), False
    >>> r = geron_sac(Bandit(), epochs=30, lr=0.5, alpha=0.05)
    >>> int(np.argmax(r["policy"][0]))
    1
    >>> bool(r["policy"][0][1] > 0.99)
    True
    >>> bool(r["entropy"][-1] < r["entropy"][0])
    True

    A hot temperature keeps the policy near-uniform (max entropy log 2):

    >>> r2 = geron_sac(Bandit(), epochs=30, lr=0.5, alpha=50.0)
    >>> bool(abs(r2["entropy"][-1] - np.log(2)) < 0.01)
    True

    References
    ----------
    Géron Ch 19
    """
    n_s, n_a = _check_env(env)
    E = int(epochs)
    T = int(steps)
    if E < 1 or T < 1:
        raise ValueError(f"geron_sac: epochs and steps must both be >= 1, got {E} and {T}")
    step_size = float(lr)
    if not (0.0 < step_size <= 1.0):
        raise ValueError(f"geron_sac: lr must lie in (0, 1], got {step_size}")
    temp = float(alpha)
    if not (np.isfinite(temp) and temp > 0):
        raise ValueError(f"geron_sac: alpha (entropy temperature) must be positive and finite, got {temp}")
    g = float(gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"geron_sac: gamma must lie in [0, 1), got {g}")

    Pi = np.full((n_s, n_a), 1.0 / n_a) if policy is None else np.asarray(policy, dtype=float)
    if Pi.shape != (n_s, n_a):
        raise ValueError(f"geron_sac: policy must have shape {(n_s, n_a)}, got {Pi.shape}")
    if np.any(Pi < 0) or not np.allclose(Pi.sum(axis=1), 1.0):
        raise ValueError("geron_sac: each policy row must be a probability distribution summing to 1")
    Q = np.zeros((n_s, n_a)) if critic is None else np.asarray(critic, dtype=float).copy()
    if Q.shape != (n_s, n_a):
        raise ValueError(f"geron_sac: critic must have shape {(n_s, n_a)}, got {Q.shape}")

    rng = int(seed) % 2**32

    def _u():
        nonlocal rng
        rng = (1664525 * rng + 1013904223) % 2**32
        return (rng + 0.5) / 2**32

    def _soft_V(P, Qt):
        logp = np.log(np.maximum(P, np.finfo(float).tiny))
        return np.sum(P * (Qt - temp * logp), axis=1)

    entropies = []
    returns = []
    s = int(env.reset())
    for _ in range(E):
        batch = []
        total = 0.0
        for _ in range(T):
            u = _u()
            a = int(np.searchsorted(np.cumsum(Pi[s]), u, side="right"))
            a = min(a, n_a - 1)
            s2, rew, done = env.step(a)
            s2, rew, done = int(s2), float(rew), bool(done)
            if not (0 <= s2 < n_s):
                raise ValueError(f"geron_sac: env.step returned state {s2} outside 0..{n_s - 1}")
            if not np.isfinite(rew):
                raise ValueError("geron_sac: env.step returned a non-finite reward")
            batch.append((s, a, rew, s2, done))
            total += rew
            s = int(env.reset()) if done else s2
        V = _soft_V(Pi, Q)
        for (bs, ba, br, bs2, bd) in batch:
            target = br + (0.0 if bd else g * V[bs2])
            Q[bs, ba] += step_size * (target - Q[bs, ba])
        Pi = np.vstack([np.asarray(geron_softmax_function(Q[i] / temp)["p"], dtype=float) for i in range(n_s)])
        logp = np.log(np.maximum(Pi, np.finfo(float).tiny))
        entropies.append(float(np.mean(-np.sum(Pi * logp, axis=1))))
        returns.append(total)

    V = _soft_V(Pi, Q)
    return RichResult(
        title="Soft actor-critic (tabular)",
        summary_lines=[
            ("States", n_s),
            ("Actions", n_a),
            ("alpha", temp),
            ("Final policy entropy", entropies[-1]),
            ("Final epoch return", returns[-1]),
        ],
        interpretation=(
            "The entropy bonus is not exploration noise bolted on: it is inside the value function, so "
            "the optimal policy is a Boltzmann distribution over Q at temperature alpha."
        ),
        payload={
            "policy": Pi,
            "Q": Q,
            "V": V,
            "entropy": np.asarray(entropies, dtype=float),
            "returns": np.asarray(returns, dtype=float),
            "alpha": temp,
            "gamma": g,
            "estimate": float(returns[-1]),
            "n": int(E * T),
            "method": "Tabular soft actor-critic: soft value backup + closed-form Boltzmann policy improvement",
        },
    )


def cheatsheet():
    return "hmsac: Soft actor-critic (SAC): entropy-regularized max reward"
