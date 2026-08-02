# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reinforcement learning: evaluate a policy by rollouts."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_reinforcement_learning"]


def _bind(env):
    """Accept an object with reset/step or a mapping of the same callables."""
    if hasattr(env, "get") and not hasattr(env, "reset"):
        reset, step = env.get("reset"), env.get("step")
    else:
        reset, step = getattr(env, "reset", None), getattr(env, "step", None)
    if not callable(reset) or not callable(step):
        raise ValueError("geron_reinforcement_learning: env must provide callable reset() and step(action)")
    return reset, step


def geron_reinforcement_learning(env, pi, gamma=0.99, n_episodes=1, max_steps=1000, seed=0):
    """
    Reinforcement learning: an agent maximising cumulative reward under a policy.

    Formula: max_pi E_pi [sum_t gamma^t r_t]

    This is the evaluation half of the loop: run the policy in the
    environment and measure the discounted return it earns. The discount
    is not a detail. gamma < 1 makes the sum finite on a
    non-terminating task and sets the agent's horizon: rewards beyond
    about 1/(1-gamma) steps contribute almost nothing, so a
    short-sighted gamma will not learn a long-delayed payoff no matter
    how good the algorithm is. That effective horizon is reported.

    ``env`` provides ``reset() -> state`` and
    ``step(action) -> (state, reward, done)`` (a 4-tuple with an info
    field is accepted). ``pi`` maps a state to an action, or to action
    probabilities that are sampled from a reproducible integer LCG.

    Parameters
    ----------
    env : object or mapping
        As described.
    pi : callable
        ``pi(state) -> action`` or ``-> probabilities``.
    gamma : float, default 0.99
        Discount in [0, 1].
    n_episodes : int, default 1
    max_steps : int, default 1000
        Cap per episode; hitting it is reported as truncation.
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: mean_return, returns, lengths, effective_horizon,
        truncated, estimate, n, method.

    Examples
    --------
    An environment paying 1 for three steps, at gamma = 0.5:
    1 + 0.5 + 0.25 = 1.75.

    >>> clock = {"t": 0}
    >>> def reset():
    ...     clock["t"] = 0
    ...     return 0
    >>> def step(a):
    ...     clock["t"] += 1
    ...     return clock["t"], 1.0, clock["t"] >= 3
    >>> r = geron_reinforcement_learning({"reset": reset, "step": step},
    ...                                  lambda s: 0, gamma=0.5)
    >>> float(r["mean_return"]), [int(v) for v in r["lengths"]]
    (1.75, [3])

    Undiscounted, the same episode is worth 3:

    >>> float(geron_reinforcement_learning({"reset": reset, "step": step},
    ...                                    lambda s: 0, gamma=1.0)["mean_return"])
    3.0

    References
    ----------
    Geron Ch 1
    """
    reset, step = _bind(env)
    if not callable(pi):
        raise ValueError("geron_reinforcement_learning: pi must be callable")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_reinforcement_learning: gamma must lie in [0, 1], got {gamma!r}")
    E = int(n_episodes)
    if E < 1:
        raise ValueError(f"geron_reinforcement_learning: n_episodes must be >= 1, got {n_episodes!r}")
    T = int(max_steps)
    if T < 1:
        raise ValueError(f"geron_reinforcement_learning: max_steps must be >= 1, got {max_steps!r}")

    s_rng = int(seed) % 2**32
    returns = np.empty(E)
    lengths = np.empty(E, dtype=int)
    truncated = 0
    for e in range(E):
        state = reset()
        total = 0.0
        disc = 1.0
        t = 0
        done = False
        while t < T and not done:
            out = pi(state)
            arr = np.atleast_1d(np.asarray(out, dtype=float))
            if np.ndim(out) == 0:
                action = out
            else:
                if np.any(arr < 0) or not np.isclose(float(arr.sum()), 1.0, atol=1e-8):
                    raise ValueError("geron_reinforcement_learning: pi returned neither an action nor a probability vector")
                s_rng = (1664525 * s_rng + 1013904223) % 2**32
                u = (s_rng + 0.5) / 2**32
                action = int(min(np.searchsorted(np.cumsum(arr), u), arr.size - 1))
            res = step(action)
            try:
                if len(res) == 3:
                    state, reward, done = res
                elif len(res) == 4:
                    state, reward, done, _ = res
                else:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(
                    "geron_reinforcement_learning: step(action) must return (state, reward, done) or a 4-tuple"
                ) from None
            r = float(reward)
            if not np.isfinite(r):
                raise ValueError("geron_reinforcement_learning: the environment returned a non-finite reward")
            total += disc * r
            disc *= g
            t += 1
        if not done:
            truncated += 1
        returns[e] = total
        lengths[e] = t

    horizon = float("inf") if g >= 1.0 else 1.0 / (1.0 - g)
    return RichResult(
        title="Policy evaluation by rollout",
        summary_lines=[("Episodes", E), ("Mean return", float(returns.mean())), ("Effective horizon", horizon)],
        interpretation="gamma sets the horizon ~1/(1-gamma); a payoff past it is invisible to the agent.",
        payload={
            "mean_return": float(returns.mean()),
            "returns": returns,
            "lengths": lengths,
            "se": float(returns.std(ddof=1) / np.sqrt(E)) if E > 1 else float("nan"),
            "effective_horizon": horizon,
            "truncated": int(truncated),
            "estimate": float(returns.mean()),
            "n": int(E),
            "method": "Monte-Carlo policy evaluation of the discounted return",
        },
    )


def cheatsheet():
    return "hmrl: Reinforcement learning, discounted return under a policy"
