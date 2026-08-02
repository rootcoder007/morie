# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Asynchronous advantage actor-critic (A3C)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_a3c"]


class _Rng:
    def __init__(self, seed):
        self.s = int(seed) % 2**32

    def uniform(self):
        self.s = (1664525 * self.s + 1013904223) % 2**32
        return (self.s + 0.5) / 2**32


def _env_fns(env, worker):
    if callable(env) and not hasattr(env, "reset"):
        env = env(worker)
    if isinstance(env, dict):
        reset, step = env.get("reset"), env.get("step")
    else:
        reset, step = getattr(env, "reset", None), getattr(env, "step", None)
    if not callable(reset) or not callable(step):
        raise ValueError(
            "geron_a3c: env must provide callable reset() and step(action), or be a factory env(worker_id)"
        )
    return reset, step


def _softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def geron_a3c(env, actor, critic, n_workers=4, lr=0.1, epochs=50, gamma=0.99, critic_lr=None, max_steps=200, seed=0):
    """
    Asynchronous advantage actor-critic (A3C).

    Formula: parallel actors asynchronously update shared parameters

    Each worker holds its own environment and its own LCG stream, pulls the
    shared parameters, rolls out one episode, computes A_t = G_t - V(s_t),
    and pushes its gradient straight into the shared parameters -- so a
    worker's gradient is generally computed against a slightly stale copy,
    which is exactly the "asynchronous" part and the reason A3C needs no
    replay buffer.

    ponytail: workers are stepped in a deterministic round-robin rather than
    on real threads. The staleness pattern is reproduced (each worker's
    gradient is applied against parameters other workers have since moved),
    the wall-clock parallelism is not.

    Parameters
    ----------
    env : object, dict, or callable
        A single environment shared by all workers, or a factory
        ``env(worker_id)`` returning one per worker.
    actor : array-like, shape (n_actions, n_features)
    critic : array-like, shape (n_features,)
    n_workers : int
        Number of parallel actors (>= 1).
    lr : float
        Actor step size (positive).
    epochs : int
        Episodes per worker (>= 1).
    gamma : float
        Discount in [0, 1].
    critic_lr : float, optional
        Critic step size; defaults to `lr`.
    max_steps : int
        Episode length cap.
    seed : int
        Base LCG seed; worker w uses seed + 7919*w.

    Returns
    -------
    result : RichResult
        Keys: actor, critic, returns, worker_returns, policy, updates,
        estimate, n, method.

    Examples
    --------
    >>> env = {"reset": lambda: [1.0], "step": lambda a: ([1.0], 1.0 if a == 0 else 0.0, True)}
    >>> r = geron_a3c(env, [[0.0], [0.0]], [0.0], n_workers=4, epochs=60, lr=0.5, seed=3)
    >>> bool(r["policy"]([1.0])[0] > 0.9)
    True
    >>> r["worker_returns"].shape
    (4, 60)
    >>> r["updates"]
    240

    Workers see different action streams, so their episode returns are not
    identical:

    >>> bool((r["worker_returns"][0] != r["worker_returns"][1]).any())
    True

    References
    ----------
    Géron Ch 19
    """
    A = np.asarray(actor, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"geron_a3c: actor must be 2-D (n_actions, n_features), got ndim={A.ndim}")
    n_actions, n_feat = A.shape
    if n_actions < 2:
        raise ValueError("geron_a3c: actor must define at least 2 actions")
    V = np.asarray(critic, dtype=float).ravel()
    if V.size != n_feat:
        raise ValueError(f"geron_a3c: critic has {V.size} weights but actor expects {n_feat} features")
    W = int(n_workers)
    if W < 1:
        raise ValueError("geron_a3c: n_workers must be >= 1")
    E = int(epochs)
    if E < 1:
        raise ValueError("geron_a3c: epochs must be >= 1")
    alpha = float(lr)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("geron_a3c: lr must be a positive finite step size")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_a3c: gamma must lie in [0, 1], got {g}")
    beta = alpha if critic_lr is None else float(critic_lr)
    if beta <= 0:
        raise ValueError("geron_a3c: critic_lr must be positive")
    M = int(max_steps)
    if M < 1:
        raise ValueError("geron_a3c: max_steps must be >= 1")

    envs = [_env_fns(env, w) for w in range(W)]
    rngs = [_Rng(seed + 7919 * w) for w in range(W)]
    shared_A = A.copy()
    shared_V = V.copy()
    worker_returns = np.zeros((W, E))
    updates = 0

    for ep in range(E):
        for w in range(W):
            reset, step = envs[w]
            rng = rngs[w]
            # Local copy == the worker's pull of the shared parameters.
            localA = shared_A.copy()
            localV = shared_V.copy()
            states, acts, rewards = [], [], []
            s = np.asarray(reset(), dtype=float).ravel()
            if s.size != n_feat:
                raise ValueError(f"geron_a3c: env.reset() returned {s.size} features but actor expects {n_feat}")
            for _ in range(M):
                probs = _softmax(localA @ s)
                a = int(np.searchsorted(np.cumsum(probs), rng.uniform(), side="right"))
                a = min(a, n_actions - 1)
                out = step(a)
                if not isinstance(out, tuple) or len(out) != 3:
                    raise ValueError("geron_a3c: env.step(action) must return a (state, reward, done) triple")
                s2, r, done = out
                s2 = np.asarray(s2, dtype=float).ravel()
                if s2.size != n_feat:
                    raise ValueError(f"geron_a3c: env.step() returned {s2.size} features but actor expects {n_feat}")
                states.append(s)
                acts.append(a)
                rewards.append(float(r))
                s = s2
                if done:
                    break
            if not states:
                raise ValueError("geron_a3c: env produced an empty episode")
            S = np.array(states)
            G = np.zeros(len(rewards))
            acc = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                acc = rewards[t] + g * acc
                G[t] = acc
            adv = G - S @ localV
            gradA = np.zeros_like(localA)
            for t in range(len(acts)):
                probs = _softmax(localA @ S[t])
                onehot = np.zeros(n_actions)
                onehot[acts[t]] = 1.0
                gradA += adv[t] * np.outer(onehot - probs, S[t])
            # Push into the shared parameters, which other workers may have
            # already moved since this worker pulled them.
            shared_A = shared_A + alpha * gradA / len(acts)
            shared_V = shared_V + beta * (S.T @ adv) / len(acts)
            worker_returns[w, ep] = float(np.sum(rewards))
            updates += 1

    def policy(state, _A=shared_A, _n=n_feat):
        st = np.asarray(state, dtype=float).ravel()
        if st.size != _n:
            raise ValueError(f"policy: expected {_n} features, got {st.size}")
        return _softmax(_A @ st)

    returns = worker_returns.mean(axis=0)

    return RichResult(
        title="Asynchronous advantage actor-critic (A3C)",
        summary_lines=[("Workers", W), ("Episodes per worker", E), ("Parameter updates", updates)],
        payload={
            "actor": shared_A,
            "critic": shared_V,
            "returns": returns,
            "worker_returns": worker_returns,
            "policy": policy,
            "updates": updates,
            "estimate": float(returns[-1]),
            "n": int(W * E),
            "method": "A3C: round-robin workers pushing advantage gradients into shared parameters",
        },
    )


def cheatsheet():
    return "hma3c: Asynchronous advantage actor-critic (A3C)"
