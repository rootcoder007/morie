# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advantage actor-critic (A2C)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_a2c"]


class _Rng:
    """Exact-integer LCG so every run is reproducible on every machine."""

    def __init__(self, seed):
        self.s = int(seed) % 2**32

    def uniform(self):
        self.s = (1664525 * self.s + 1013904223) % 2**32
        return (self.s + 0.5) / 2**32


def _env_fns(env):
    if isinstance(env, dict):
        reset, step = env.get("reset"), env.get("step")
    else:
        reset, step = getattr(env, "reset", None), getattr(env, "step", None)
    if not callable(reset) or not callable(step):
        raise ValueError("geron_a2c: env must provide callable reset() and step(action)")
    return reset, step


def _softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def _rollout(reset, step, actor, critic, rng, gamma, max_steps, n_feat):
    states, actions, rewards = [], [], []
    s = np.asarray(reset(), dtype=float).ravel()
    if s.size != n_feat:
        raise ValueError(f"geron_a2c: env.reset() returned {s.size} features but actor expects {n_feat}")
    for _ in range(max_steps):
        probs = _softmax(actor @ s)
        u = rng.uniform()
        a = int(np.searchsorted(np.cumsum(probs), u, side="right"))
        a = min(a, probs.size - 1)
        out = step(a)
        if not isinstance(out, tuple) or len(out) != 3:
            raise ValueError("geron_a2c: env.step(action) must return a (state, reward, done) triple")
        s2, r, done = out
        s2 = np.asarray(s2, dtype=float).ravel()
        if s2.size != n_feat:
            raise ValueError(f"geron_a2c: env.step() returned {s2.size} features but actor expects {n_feat}")
        states.append(s)
        actions.append(a)
        rewards.append(float(r))
        s = s2
        if done:
            break
    G = np.zeros(len(rewards))
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        G[t] = acc
    return np.array(states), np.array(actions), np.array(rewards), G


def geron_a2c(env, actor, critic, epochs=100, lr=0.1, gamma=0.99, critic_lr=None, max_steps=200, seed=0):
    """
    Advantage actor-critic (A2C).

    Formula: policy update uses A(s,a) = Q(s,a) - V(s)

    Linear softmax policy and linear value baseline, trained by Monte-Carlo
    advantage actor-critic. `actor` is an (n_actions, n_features) logit
    matrix and `critic` an (n_features,) value vector; the environment is any
    object (or dict) with ``reset() -> state`` and
    ``step(action) -> (state, reward, done)``.

    Per episode: A_t = G_t - V(s_t), then
    actor += lr * A_t * (onehot(a_t) - pi(.|s_t)) s_t^T and
    critic += critic_lr * A_t * s_t.

    Parameters
    ----------
    env : object or dict
        Environment with reset/step as described.
    actor : array-like, shape (n_actions, n_features)
        Initial policy logits.
    critic : array-like, shape (n_features,)
        Initial value weights.
    epochs : int
        Number of episodes (>= 1).
    lr : float
        Actor step size (positive).
    gamma : float
        Discount in [0, 1].
    critic_lr : float, optional
        Critic step size; defaults to `lr`.
    max_steps : int
        Episode length cap.
    seed : int
        LCG seed for action sampling.

    Returns
    -------
    result : RichResult
        Keys: actor, critic, returns, policy, value, advantages, estimate,
        n, method.

    Examples
    --------
    A one-step bandit where action 0 pays 1 and action 1 pays 0: the policy
    should converge onto action 0 and the value baseline onto its return.

    >>> env = {"reset": lambda: [1.0], "step": lambda a: ([1.0], 1.0 if a == 0 else 0.0, True)}
    >>> r = geron_a2c(env, [[0.0], [0.0]], [0.0], epochs=200, lr=0.5, seed=1)
    >>> bool(r["policy"]([1.0])[0] > 0.9)
    True
    >>> bool(r["returns"][-20:].mean() > r["returns"][:20].mean())
    True
    >>> bool(abs(r["value"]([1.0]) - 1.0) < 0.2)
    True

    References
    ----------
    Géron Ch 19
    """
    reset, step = _env_fns(env)
    A = np.asarray(actor, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"geron_a2c: actor must be 2-D (n_actions, n_features), got ndim={A.ndim}")
    n_actions, n_feat = A.shape
    if n_actions < 2:
        raise ValueError("geron_a2c: actor must define at least 2 actions")
    V = np.asarray(critic, dtype=float).ravel()
    if V.size != n_feat:
        raise ValueError(f"geron_a2c: critic has {V.size} weights but actor expects {n_feat} features")
    E = int(epochs)
    if E < 1:
        raise ValueError("geron_a2c: epochs must be >= 1")
    alpha = float(lr)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("geron_a2c: lr must be a positive finite step size")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_a2c: gamma must lie in [0, 1], got {g}")
    beta = alpha if critic_lr is None else float(critic_lr)
    if beta <= 0:
        raise ValueError("geron_a2c: critic_lr must be positive")
    if int(max_steps) < 1:
        raise ValueError("geron_a2c: max_steps must be >= 1")

    A = A.copy()
    V = V.copy()
    rng = _Rng(seed)
    ep_returns = np.zeros(E)
    last_adv = np.zeros(0)

    for ep in range(E):
        S, acts, rew, G = _rollout(reset, step, A, V, rng, g, int(max_steps), n_feat)
        if S.size == 0:
            raise ValueError("geron_a2c: env produced an empty episode")
        ep_returns[ep] = float(np.sum(rew))
        baseline = S @ V
        adv = G - baseline
        last_adv = adv
        gradA = np.zeros_like(A)
        for t in range(len(acts)):
            probs = _softmax(A @ S[t])
            onehot = np.zeros(n_actions)
            onehot[acts[t]] = 1.0
            gradA += adv[t] * np.outer(onehot - probs, S[t])
        A = A + alpha * gradA / len(acts)
        V = V + beta * (S.T @ adv) / len(acts)

    def policy(state, _A=A, _n=n_feat):
        s = np.asarray(state, dtype=float).ravel()
        if s.size != _n:
            raise ValueError(f"policy: expected {_n} features, got {s.size}")
        return _softmax(_A @ s)

    def value(state, _V=V, _n=n_feat):
        s = np.asarray(state, dtype=float).ravel()
        if s.size != _n:
            raise ValueError(f"value: expected {_n} features, got {s.size}")
        return float(_V @ s)

    return RichResult(
        title="Advantage actor-critic (A2C)",
        summary_lines=[("Episodes", E), ("Mean return (last 10%)", float(np.mean(ep_returns[-max(1, E // 10) :])))],
        payload={
            "actor": A,
            "critic": V,
            "returns": ep_returns,
            "policy": policy,
            "value": value,
            "advantages": last_adv,
            "estimate": float(ep_returns[-1]),
            "n": int(E),
            "method": "A2C with a linear softmax policy and a linear value baseline",
        },
    )


def cheatsheet():
    return "hma2c: Advantage actor-critic (A2C)"
