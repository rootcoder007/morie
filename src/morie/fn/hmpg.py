# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Policy gradient (REINFORCE) estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_policy_gradient"]


def _steps(ep, e):
    out = []
    for t, st in enumerate(ep):
        if hasattr(st, "get"):
            if "action" not in st or "reward" not in st:
                raise ValueError(f"geron_policy_gradient: step {t} of episode {e} needs 'action' and 'reward'")
            out.append((st.get("state"), st["action"], float(st["reward"])))
        else:
            try:
                s, a, r = st
            except (TypeError, ValueError):
                raise ValueError(
                    f"geron_policy_gradient: step {t} of episode {e} must be (state, action, reward) or a mapping"
                ) from None
            out.append((s, a, float(r)))
    return out


def geron_policy_gradient(trajectories, policy, gamma=0.99, baseline=False):
    """
    Policy gradient (REINFORCE) update.

    Formula: grad J = E[grad log pi(a|s; theta) * Q(s,a)]

    Every action in an episode is credited with the RETURN that followed
    it, so an action is reinforced for everything that happened after it
    -- including luck. That is why the estimator is unbiased and very
    noisy, and why the returns are discounted and (optionally) centred by
    a baseline: subtracting any function of the state leaves the
    expectation untouched and can only cut the variance.

    ``policy(state, action)`` must return the gradient of log pi at that
    pair, or a ``(log_prob, gradient)`` pair.

    Parameters
    ----------
    trajectories : sequence of sequences
        Episodes of ``(state, action, reward)`` steps or mappings with
        those keys.
    policy : callable
        As described.
    gamma : float, default 0.99
        Discount in [0, 1].
    baseline : bool, default False
        Subtract the mean return over all steps.

    Returns
    -------
    result : RichResult
        Keys: gradient, returns, mean_return, n_steps, baseline_value,
        estimate, n, method.

    Examples
    --------
    One step, reward 2, gradient (1, 0):

    >>> g = lambda s, a: np.array([1.0, 0.0])
    >>> r = geron_policy_gradient([[(0, 0, 2.0)]], g, gamma=1.0)
    >>> [float(v) for v in r["gradient"]]
    [2.0, 0.0]

    Two steps at gamma = 0.5 with rewards 1 and 1: the returns are 1.5
    and 1, and each is paired with that step's own gradient.

    >>> gs = lambda s, a: (np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0]))
    >>> r2 = geron_policy_gradient([[(0, 0, 1.0), (1, 1, 1.0)]], gs, gamma=0.5)
    >>> [float(v) for v in r2["returns"]]
    [1.5, 1.0]
    >>> [float(v) for v in r2["gradient"]]
    [1.5, 1.0]

    A baseline centres the returns without changing their order:

    >>> b = geron_policy_gradient([[(0, 0, 1.0), (1, 1, 1.0)]], gs, gamma=0.5, baseline=True)
    >>> [float(v) for v in b["gradient"]]
    [0.25, -0.25]

    References
    ----------
    Geron Ch 19
    """
    if not callable(policy):
        raise ValueError("geron_policy_gradient: policy must be callable")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_policy_gradient: gamma must lie in [0, 1], got {gamma!r}")
    eps = list(trajectories)
    if not eps:
        raise ValueError("geron_policy_gradient: trajectories is empty")

    all_steps = []
    all_returns = []
    for e, ep in enumerate(eps):
        steps = _steps(list(ep), e)
        if not steps:
            raise ValueError(f"geron_policy_gradient: episode {e} has no steps")
        G = 0.0
        rets = np.empty(len(steps))
        for t in range(len(steps) - 1, -1, -1):
            G = steps[t][2] + g * G
            rets[t] = G
        all_steps.extend(steps)
        all_returns.append(rets)

    returns = np.concatenate(all_returns)
    b = float(np.mean(returns)) if baseline else 0.0

    grad = None
    for (s, a, _), G in zip(all_steps, returns):
        out = policy(s, a)
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError("geron_policy_gradient: policy returned a tuple that is not (log_prob, gradient)")
            out = out[1]
        gv = np.atleast_1d(np.asarray(out, dtype=float)).ravel()
        if not np.all(np.isfinite(gv)):
            raise ValueError(f"geron_policy_gradient: policy returned a non-finite gradient at state {s!r}, action {a!r}")
        if grad is None:
            grad = np.zeros_like(gv)
        elif gv.shape != grad.shape:
            raise ValueError(f"geron_policy_gradient: policy gradients change shape ({grad.shape} then {gv.shape})")
        grad = grad + gv * (G - b)

    grad = grad / len(eps)
    return RichResult(
        title="Policy gradient (REINFORCE)",
        summary_lines=[("Episodes", len(eps)), ("Steps", int(returns.size)), ("Mean return", float(returns.mean()))],
        interpretation="Every action is credited with all the luck that followed; a baseline only cuts the variance.",
        payload={
            "gradient": grad,
            "returns": returns,
            "mean_return": float(returns.mean()),
            "n_steps": int(returns.size),
            "n_episodes": len(eps),
            "baseline_value": b,
            "estimate": grad,
            "n": int(returns.size),
            "method": "REINFORCE gradient with discounted returns" + (" and a mean baseline" if baseline else ""),
        },
    )


def cheatsheet():
    return "hmpg: Policy gradient (REINFORCE) estimator"
