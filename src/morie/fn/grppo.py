# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proximal Policy Optimization clipped surrogate objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ppo_clipped_objective"]

_METHOD = "PPO clipped surrogate objective"


def geron_ppo_clipped_objective(ratios, advantages, eps=0.2):
    r"""PPO's pessimistic surrogate.

    .. math::
        L = \mathbb{E}_t\Bigl[\min\bigl(r_t A_t,\;
            \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\bigr)\Bigr],
        \qquad r_t = \frac{\pi_{\theta}}{\pi_{\text{old}}}

    The ``min`` -- not the clip alone -- is what makes this a lower
    bound.  For a positive advantage it caps the gain once
    :math:`r > 1+\epsilon`, so there is nothing to gain by moving further;
    for a negative advantage the *unclipped* term is the smaller one when
    :math:`r < 1-\epsilon`, which keeps a gradient that pulls a bad action
    back rather than letting it run away.  Getting only the clip and
    dropping the min removes that safety and is the classic PPO bug, so
    both branches are reported per step.

    Parameters
    ----------
    ratios : array-like, shape (T,)
        Probability ratios; must be positive.
    advantages : array-like, shape (T,)
    eps : float, optional
        Clip range, positive and below 1 (PPO uses 0.2).

    Returns
    -------
    RichResult
        Payload keys ``objective`` (mean), ``per_step``, ``unclipped``,
        ``clipped``, ``clipped_fraction``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 19, PPO section.

    Examples
    --------
    A positive advantage with ``r = 1.5`` is capped at ``1.2 * 1 = 1.2``;
    a negative advantage with ``r = 0.5`` keeps the unclipped
    ``0.5 * -1 = -0.5``, which is smaller than the clipped ``-0.8``:

    >>> r = geron_ppo_clipped_objective([1.5, 0.5], [1.0, -1.0], eps=0.2)
    >>> r["per_step"]
    [1.2, -0.8]
    >>> round(r["objective"], 10)
    0.2

    Inside the trust region nothing is clipped:

    >>> geron_ppo_clipped_objective([1.0], [3.0])["clipped_fraction"]
    0.0
    """
    r = np.asarray(ratios, dtype=float).ravel()
    A = np.asarray(advantages, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("ratios is empty.")
    if r.shape != A.shape:
        raise ValueError(f"ratios has {r.size} entries but advantages has {A.size}.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(A)):
        raise ValueError("ratios and advantages must be finite.")
    if np.any(r <= 0):
        raise ValueError(
            f"probability ratios must be positive, got minimum {float(r.min())}."
        )
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must lie in (0, 1), got {eps}.")

    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    per = np.minimum(unclipped, clipped)
    frac = float(np.mean(~np.isclose(unclipped, clipped)))

    return RichResult(
        title="PPO clipped objective",
        summary_lines=[("Objective", float(per.mean())), ("Clipped fraction", frac)],
        payload={
            "objective": float(per.mean()),
            "per_step": per.tolist(),
            "unclipped": unclipped.tolist(),
            "clipped": clipped.tolist(),
            "clipped_fraction": frac,
            "estimate": float(per.mean()),
            "n": int(r.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grppo: L = mean min(rA, clip(r,1-e,1+e)A); the min, not the clip, makes it a lower bound"
