# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct preference optimization (DPO)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dpo"]


def _logsigmoid(z):
    # -log(1+exp(-z)) without overflow for either sign.
    return np.where(z >= 0, -np.log1p(np.exp(-np.abs(z))), z - np.log1p(np.exp(-np.abs(z))))


def geron_dpo(pi, pi_ref, preferences=None, beta=0.1):
    """
    Direct preference optimization (DPO).

    Formula: L = -log sigmoid(beta * log(pi(y_w|x)/pi_ref(y_w|x))
    - beta * log(pi(y_l|x)/pi_ref(y_l|x)))

    ``pi`` and ``pi_ref`` hold the policy and reference **log**
    probabilities of the two candidate completions per prompt, shape
    ``(B, 2)``. ``preferences`` says which column is the winner (0 or 1);
    if omitted, column 0 is taken as the chosen response, DPO's usual
    ``(chosen, rejected)`` convention.

    The implicit reward ``beta * log(pi/pi_ref)`` is returned per
    completion, together with the reward margin whose sigmoid is exactly
    the modelled preference probability. A policy identical to the
    reference has zero margin and loss ``log 2``.

    Parameters
    ----------
    pi : array-like, shape (B, 2)
        Log-probabilities under the policy being trained.
    pi_ref : array-like, shape (B, 2)
        Log-probabilities under the frozen reference policy.
    preferences : array-like of int, shape (B,), optional
        Index (0 or 1) of the preferred completion. Default all zeros.
    beta : float, default 0.1
        Inverse temperature on the implicit reward; positive.

    Returns
    -------
    result : RichResult
        Keys: loss, per_pair_loss, margin, reward_chosen,
        reward_rejected, accuracy, prob_preferred, estimate, n, method.

    Examples
    --------
    A policy equal to its reference is indifferent, so the loss is
    ``log 2`` and the modelled preference probability is one half:

    >>> import math
    >>> r = geron_dpo([[-1.0, -2.0]], [[-1.0, -2.0]])
    >>> round(r["loss"], 9) == round(math.log(2), 9)
    True
    >>> round(r["prob_preferred"][0], 9)
    0.5

    Raising the winner's log-prob by 1 nat at beta = 1 gives a margin of
    1 and loss ``log(1 + e^-1)``:

    >>> r2 = geron_dpo([[0.0, -2.0]], [[-1.0, -2.0]], beta=1.0)
    >>> round(r2["margin"][0], 9)
    1.0
    >>> round(r2["loss"], 6)
    0.313262
    >>> r2["accuracy"]
    1.0

    References
    ----------
    Géron Ch 15
    """
    lp = np.atleast_2d(np.asarray(pi, dtype=float))
    lr = np.atleast_2d(np.asarray(pi_ref, dtype=float))
    if lp.shape != lr.shape:
        raise ValueError(f"geron_dpo: pi has shape {lp.shape} but pi_ref has shape {lr.shape}")
    if lp.ndim != 2 or lp.shape[1] != 2:
        raise ValueError(f"geron_dpo: pi must have shape (B, 2) -- chosen and rejected -- got {lp.shape}")
    if lp.size == 0:
        raise ValueError("geron_dpo: no preference pairs supplied")
    if not np.all(np.isfinite(lp)) or not np.all(np.isfinite(lr)):
        raise ValueError("geron_dpo: log-probabilities must be finite")
    if np.any(lp > 0) or np.any(lr > 0):
        raise ValueError("geron_dpo: pi and pi_ref must be LOG probabilities (<= 0)")
    b = float(beta)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"geron_dpo: beta must be positive and finite, got {beta!r}")

    B = lp.shape[0]
    if preferences is None:
        pref = np.zeros(B, dtype=int)
    else:
        pref = np.asarray(preferences).ravel()
        if pref.size != B:
            raise ValueError(f"geron_dpo: preferences has {pref.size} entries but there are {B} pairs")
        pref = pref.astype(int)
        if not np.all((pref == 0) | (pref == 1)):
            raise ValueError("geron_dpo: preferences must be 0 or 1 (which column won)")

    rows = np.arange(B)
    win, lose = pref, 1 - pref
    rw = b * (lp[rows, win] - lr[rows, win])
    rl = b * (lp[rows, lose] - lr[rows, lose])
    margin = rw - rl
    per = -_logsigmoid(margin)
    loss = float(np.mean(per))
    prob = 1.0 / (1.0 + np.exp(-margin))

    return RichResult(
        title="DPO loss",
        summary_lines=[("Loss", loss), ("Mean margin", float(np.mean(margin))), ("beta", b)],
        interpretation="Loss log 2 means the policy is indifferent; below it, the winner is preferred.",
        payload={
            "loss": loss,
            "per_pair_loss": per.tolist(),
            "margin": margin.tolist(),
            "reward_chosen": rw.tolist(),
            "reward_rejected": rl.tolist(),
            "prob_preferred": prob.tolist(),
            "accuracy": float(np.mean(margin > 0)),
            "beta": b,
            "estimate": loss,
            "n": int(B),
            "method": "DPO loss -log sigmoid(beta*(log pi/pi_ref)_w - beta*(log pi/pi_ref)_l)",
        },
    )


def cheatsheet():
    return "hmdpo: Direct preference optimization (DPO)"


# compact alias per ledger/NAMING.md
gerondpo = geron_dpo
