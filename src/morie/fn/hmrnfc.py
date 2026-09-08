# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""REINFORCE parameter update from sampled episodes."""

from . import _array_core as np

from ._richresult import RichResult
from .hmpg import geron_policy_gradient

__all__ = ["geron_reinforce"]


def geron_reinforce(episodes, policy, gamma=0.99, eta=0.01, theta=None, baseline=True):
    """
    REINFORCE algorithm: sample trajectories, update theta by the advantage.

    Formula: theta <- theta + eta * sum_t grad log pi(a_t|s_t) * G_t

    The gradient itself is DELEGATED to
    :func:`~morie.fn.hmpg.geron_policy_gradient`; what this adds is the
    ASCENT step, and the discipline around it. The update is on-policy:
    once theta moves, the episodes that produced this gradient are stale
    and must be thrown away, which is what makes REINFORCE so
    sample-hungry.

    The baseline is on by default here, unlike in the raw gradient,
    because without it the update size scales with the arbitrary zero of
    the reward -- add 100 to every reward and an unbaselined REINFORCE
    reinforces everything.

    Parameters
    ----------
    episodes : sequence of sequences
        ``(state, action, reward)`` steps per episode.
    policy : callable
        ``policy(state, action) -> grad log pi`` (or ``(logp, grad)``).
    gamma : float, default 0.99
    eta : float, default 0.01
        Step size (positive).
    theta : array-like, optional
        Current parameters; default zeros.
    baseline : bool, default True
        Centre the returns.

    Returns
    -------
    result : RichResult
        Keys: theta, step, gradient, returns, mean_return, estimate, n,
        method.

    Examples
    --------
    Two steps at gamma = 0.5, rewards 1 and 1, gradients (1,0) and (0,1):
    the centred returns are +0.25 and -0.25, so the step is eta times
    that.

    >>> gs = lambda s, a: (np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0]))
    >>> r = geron_reinforce([[(0, 0, 1.0), (1, 1, 1.0)]], gs, gamma=0.5, eta=0.1)
    >>> [float(v) for v in r["step"]]
    [0.025, -0.025]
    >>> [float(v) for v in r["theta"]]
    [0.025, -0.025]

    Without the baseline the raw returns drive the step:

    >>> [float(v) for v in geron_reinforce([[(0, 0, 1.0), (1, 1, 1.0)]], gs,
    ...                                    gamma=0.5, eta=0.1, baseline=False)["step"]]
    [0.15000000000000002, 0.1]

    References
    ----------
    Geron Ch 19
    """
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_reinforce: eta must be positive and finite, got {eta!r}")
    base = geron_policy_gradient(episodes, policy, gamma=gamma, baseline=baseline)
    grad = np.asarray(base["gradient"], dtype=float)
    th = np.zeros_like(grad) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float)).astype(float)
    if th.shape != grad.shape:
        raise ValueError(f"geron_reinforce: theta has shape {th.shape} but the gradient has shape {grad.shape}")

    step = lr * grad
    theta_next = th + step
    return RichResult(
        title="REINFORCE update",
        summary_lines=[("Episodes", int(base["n_episodes"])), ("Mean return", float(base["mean_return"])), ("Step norm", float(np.linalg.norm(step)))],
        interpretation="On-policy: after this step the episodes that produced the gradient are stale.",
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "gradient": grad,
            "returns": base["returns"],
            "mean_return": float(base["mean_return"]),
            "baseline_value": float(base["baseline_value"]),
            "estimate": theta_next,
            "n": int(base["n"]),
            "method": "REINFORCE ascent step on the gradient from morie.fn.hmpg",
        },
    )


def cheatsheet():
    return "hmrnfc: REINFORCE ascent step from sampled episodes"


# compact alias per ledger/NAMING.md
geronreinforce = geron_reinforce
