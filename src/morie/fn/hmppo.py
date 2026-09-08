# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proximal policy optimization with the clipped surrogate objective."""

from . import _array_core as np

from ._richresult import RichResult
from .hmrl import _bind

__all__ = ["geron_ppo"]


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def geron_ppo(env, policy, epochs=20, lr=0.1, clip_eps=0.2, gamma=0.99,
              n_episodes=8, max_steps=50, n_updates=4, seed=0):
    """
    Proximal policy optimization (PPO) clipped-surrogate objective.

    Formula: L = E[min(r_t(theta) A_t, clip(r_t, 1-e, 1+e) A_t)]

    The clip is a one-sided brake. Where the advantage is positive the
    objective stops improving once the probability ratio passes 1+e, so
    there is no gradient left to push the update further; where it is
    negative the same happens below 1-e. Taking the MINIMUM of the
    clipped and unclipped terms is what makes it a pessimistic bound
    rather than a mere clamp -- it never lets the surrogate report an
    improvement the unclipped objective does not support.

    That is the entire mechanism by which PPO stays near the behaviour
    policy without the trust-region machinery of TRPO, and it is why the
    same batch can be reused for several gradient steps: ``n_updates``
    passes over one batch of episodes.

    The policy is a tabular softmax over discrete states, ``policy``
    being the (n_states, n_actions) logit matrix; ``env`` follows the
    ``reset``/``step`` contract of
    :func:`~morie.fn.hmrl.geron_reinforcement_learning`.

    Parameters
    ----------
    env : object or mapping
        ``reset() -> state``, ``step(a) -> (state, reward, done)``.
    policy : array-like, shape (n_states, n_actions)
        Initial logits.
    epochs : int, default 20
        Collect-and-update rounds.
    lr : float, default 0.1
        Ascent step (positive).
    clip_eps : float, default 0.2
        Clipping half-width in (0, 1).
    gamma : float, default 0.99
    n_episodes : int, default 8
        Episodes per batch.
    max_steps : int, default 50
    n_updates : int, default 4
        Gradient steps per batch.
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: theta, probabilities, return_history, surrogate_history,
        clip_fraction, estimate, n, method.

    Examples
    --------
    A one-state bandit paying the action index: action 1 is worth 1 and
    action 0 nothing, so its probability must rise from 1/2.

    >>> def reset():
    ...     return 0
    >>> def step(a):
    ...     return 0, float(a), True
    >>> r = geron_ppo({"reset": reset, "step": step}, [[0.0, 0.0]], epochs=30, lr=0.5, seed=1)
    >>> bool(r["probabilities"][0][1] > 0.9)
    True
    >>> bool(r["return_history"][-1] >= r["return_history"][0])
    True

    The clipped fraction is a fraction:

    >>> bool(0.0 <= r["clip_fraction"] <= 1.0)
    True

    References
    ----------
    Geron Ch 19
    """
    reset, step = _bind(env)
    Z = np.atleast_2d(np.asarray(policy, dtype=float)).astype(float).copy()
    if Z.ndim != 2 or Z.size == 0:
        raise ValueError(f"geron_ppo: policy must be a non-empty (n_states, n_actions) logit matrix, got shape {Z.shape}")
    nS, nA = Z.shape
    if nA < 2:
        raise ValueError(f"geron_ppo: need at least 2 actions, got {nA}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_ppo: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_ppo: lr must be positive and finite, got {lr!r}")
    eps = float(clip_eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"geron_ppo: clip_eps must lie in (0, 1), got {clip_eps!r}")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_ppo: gamma must lie in [0, 1], got {gamma!r}")
    B = int(n_episodes)
    if B < 1:
        raise ValueError(f"geron_ppo: n_episodes must be >= 1, got {n_episodes!r}")
    U = int(n_updates)
    if U < 1:
        raise ValueError(f"geron_ppo: n_updates must be >= 1, got {n_updates!r}")
    T = int(max_steps)
    if T < 1:
        raise ValueError(f"geron_ppo: max_steps must be >= 1, got {max_steps!r}")

    rng = int(seed) % 2**32
    ret_hist, sur_hist = [], []
    clipped_total = 0
    seen_total = 0
    for _ in range(E):
        states, actions, rets = [], [], []
        ep_returns = []
        for _ep in range(B):
            s = reset()
            traj_s, traj_a, traj_r = [], [], []
            done = False
            t = 0
            while t < T and not done:
                si = int(s)
                if not (0 <= si < nS):
                    raise ValueError(f"geron_ppo: the environment returned state {si} outside the {nS} policy rows")
                p = _softmax(Z[si])
                rng = (1664525 * rng + 1013904223) % 2**32
                u = (rng + 0.5) / 2**32
                a = int(min(np.searchsorted(np.cumsum(p), u), nA - 1))
                out = step(a)
                if len(out) == 4:
                    s, rew, done, _ = out
                else:
                    s, rew, done = out
                traj_s.append(si)
                traj_a.append(a)
                traj_r.append(float(rew))
                t += 1
            G = 0.0
            gs = np.empty(len(traj_r))
            for k in range(len(traj_r) - 1, -1, -1):
                G = traj_r[k] + g * G
                gs[k] = G
            states.extend(traj_s)
            actions.extend(traj_a)
            rets.extend(gs.tolist())
            ep_returns.append(gs[0] if gs.size else 0.0)

        S = np.asarray(states, dtype=int)
        Aa = np.asarray(actions, dtype=int)
        R = np.asarray(rets, dtype=float)
        adv = R - R.mean()
        sd = R.std()
        if sd > 1e-12:
            adv = adv / sd
        old_logp = np.array([np.log(_softmax(Z[s])[a] + 1e-300) for s, a in zip(S, Aa)])

        for _u in range(U):
            grad = np.zeros_like(Z)
            sur = 0.0
            for i in range(S.size):
                p = _softmax(Z[S[i]])
                ratio = float(np.exp(np.log(p[Aa[i]] + 1e-300) - old_logp[i]))
                unclipped = ratio * adv[i]
                clipped = float(np.clip(ratio, 1 - eps, 1 + eps)) * adv[i]
                sur += min(unclipped, clipped)
                seen_total += 1
                if unclipped <= clipped:
                    dlog = -p.copy()
                    dlog[Aa[i]] += 1.0
                    grad[S[i]] += adv[i] * ratio * dlog
                else:
                    clipped_total += 1
            Z += eta * grad / max(S.size, 1)
            if _u == 0:
                sur_hist.append(sur / max(S.size, 1))
        ret_hist.append(float(np.mean(ep_returns)))

    probs = np.vstack([_softmax(Z[s]) for s in range(nS)])
    return RichResult(
        title="PPO (clipped surrogate)",
        summary_lines=[("Epochs", E), ("Final mean return", ret_hist[-1]), ("Clip fraction", clipped_total / max(seen_total, 1))],
        interpretation="The min of clipped and unclipped makes the surrogate a pessimistic bound, not just a clamp.",
        payload={
            "theta": Z,
            "probabilities": probs,
            "return_history": ret_hist,
            "surrogate_history": sur_hist,
            "clip_fraction": clipped_total / max(seen_total, 1),
            "clip_eps": eps,
            "estimate": probs,
            "n": int(E * B),
            "method": "PPO with a clipped surrogate on a tabular softmax policy",
        },
    )


def cheatsheet():
    return "hmppo: PPO clipped-surrogate policy optimization"


# compact alias per ledger/NAMING.md
geronppo = geron_ppo
