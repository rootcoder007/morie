# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""REINFORCE policy gradient update (Monte Carlo)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_reinforce_policy_gradient"]

_METHOD = "REINFORCE Monte Carlo policy gradient"


def geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha, baseline=None):
    r"""Score-function policy gradient ascent over one episode.

    .. math::
        \theta \leftarrow \theta + \alpha \sum_t G_t\,
            \nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)

    ``log_probs`` must be the *gradients* :math:`\nabla_\theta \log \pi`,
    one row per step -- the scalar log-probabilities carry no direction
    and cannot be used here.  That contract is enforced rather than
    guessed at.  Ascent, not descent: the sign is positive because we are
    maximising return.  Subtracting a baseline (the mean return is the
    usual choice) leaves the gradient unbiased -- :math:`\mathbb{E}[
    \nabla\log\pi] = 0` -- while cutting its variance, which is the
    difference between REINFORCE working and not.

    Parameters
    ----------
    theta : array-like, shape (p,)
    log_probs : array-like, shape (T, p)
        Per-step score vectors :math:`\nabla_\theta \log \pi`.
    returns_G : array-like, shape (T,)
        Discounted returns per step (see :mod:`morie.fn.grret`).
    alpha : float
        Positive step size.
    baseline : {None, "mean"} or float, optional
        Constant subtracted from every return.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``gradient``, ``advantages``,
        ``step_norm``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Policy Gradients / REINFORCE section.

    Examples
    --------
    Two steps, score vectors ``[1, 0]`` and ``[0, 1]``, returns 2 and -1:
    the gradient is ``[2, -1]`` and ``alpha = 0.5`` moves theta to
    ``[1, -0.5]``.

    >>> r = geron_reinforce_policy_gradient([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
    ...                                     [2.0, -1.0], alpha=0.5)
    >>> r["gradient"]
    [2.0, -1.0]
    >>> r["theta_new"]
    [1.0, -0.5]

    A mean baseline recentres the returns to sum zero:

    >>> b = geron_reinforce_policy_gradient([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
    ...                                     [2.0, -1.0], alpha=0.5, baseline="mean")
    >>> b["advantages"]
    [1.5, -1.5]
    """
    th = np.asarray(theta, dtype=float).ravel()
    S = np.atleast_2d(np.asarray(log_probs, dtype=float))
    G = np.asarray(returns_G, dtype=float).ravel()
    if th.size == 0:
        raise ValueError("theta is empty.")
    if S.ndim != 2:
        raise ValueError(
            f"log_probs must be a (T, p) array of score vectors grad log pi, got shape {S.shape}."
        )
    if S.shape[1] != th.size:
        raise ValueError(
            f"log_probs rows have {S.shape[1]} components but theta has {th.size}; "
            "pass gradients of log pi, not scalar log-probabilities."
        )
    if S.shape[0] != G.size:
        raise ValueError(f"log_probs has {S.shape[0]} steps but returns_G has {G.size}.")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(G)) and np.all(np.isfinite(th))):
        raise ValueError("theta, log_probs and returns_G must be finite.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be a positive finite float, got {alpha}.")

    if baseline is None:
        b = 0.0
    elif isinstance(baseline, str):
        if baseline != "mean":
            raise ValueError(f"baseline must be None, 'mean' or a float, got {baseline!r}.")
        b = float(G.mean())
    else:
        b = float(baseline)
        if not np.isfinite(b):
            raise ValueError(f"baseline must be finite, got {b}.")
    adv = G - b
    grad = S.T @ adv
    new = th + alpha * grad

    return RichResult(
        title="REINFORCE update",
        summary_lines=[("Steps", int(G.size)), ("Step norm", float(np.linalg.norm(alpha * grad)))],
        payload={
            "theta_new": new.tolist(),
            "gradient": grad.tolist(),
            "advantages": adv.tolist(),
            "baseline": b,
            "step_norm": float(np.linalg.norm(alpha * grad)),
            "estimate": new.tolist(),
            "n": int(G.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrein: theta += alpha sum_t G_t grad log pi_t (ascent); log_probs must be score VECTORS"
