# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""XGBoost split-gain formula with regularization."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_xgboost_gain"]

_METHOD = "XGBoost regularized split gain"


def geron_xgboost_gain(GL, HL, GR, HR, lam=1.0, gamma=0.0):
    r"""Gain from splitting a node into the given left/right halves.

    .. math::
        \mathrm{Gain} = \frac{1}{2}\left[
            \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}
            - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma

    Each term :math:`G^2/(H+\lambda)` is (twice) the loss reduction from
    giving a leaf its optimal weight :math:`-G/(H+\lambda)`, straight out
    of the second-order Taylor expansion -- which is why XGBoost needs
    both gradients and Hessians where plain gradient boosting needs only
    gradients.  :math:`\lambda` shrinks leaf weights; :math:`\gamma` is a
    flat toll per split, so a *negative* gain is the pruning signal and
    is returned as such rather than clamped to zero.

    Parameters
    ----------
    GL, GR : float
        Sum of gradients in each child.
    HL, HR : float
        Sum of Hessians in each child; non-negative.
    lam : float, optional
        L2 leaf-weight penalty, non-negative.
    gamma : float, optional
        Per-split complexity cost, non-negative.

    Returns
    -------
    RichResult
        Payload keys ``gain``, ``left_score``, ``right_score``,
        ``parent_score``, ``left_weight``, ``right_weight``,
        ``should_split``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 6, XGBoost section.

    Examples
    --------
    A split that separates opposite gradients is worth a lot: with
    ``lam = 1`` the children score ``4/3`` each and the parent 0.

    >>> r = geron_xgboost_gain(GL=-2.0, HL=2.0, GR=2.0, HR=2.0, lam=1.0)
    >>> round(r["left_score"], 6), round(r["parent_score"], 6)
    (1.333333, 0.0)
    >>> round(r["gain"], 6)
    1.333333

    A big enough gamma prunes it:

    >>> p = geron_xgboost_gain(-2.0, 2.0, 2.0, 2.0, lam=1.0, gamma=2.0)
    >>> round(p["gain"], 6), p["should_split"]
    (-0.666667, False)
    """
    GL, HL, GR, HR = float(GL), float(HL), float(GR), float(HR)
    for name, v in (("GL", GL), ("HL", HL), ("GR", GR), ("HR", HR)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}.")
    if HL < 0 or HR < 0:
        raise ValueError(f"Hessian sums must be non-negative, got HL={HL}, HR={HR}.")
    lam = float(lam)
    gamma = float(gamma)
    if lam < 0 or gamma < 0:
        raise ValueError(f"lam and gamma must be non-negative, got {lam} and {gamma}.")
    for label, denom in (("left", HL + lam), ("right", HR + lam),
                         ("parent", HL + HR + lam)):
        if denom <= 0:
            raise ValueError(
                f"the {label} denominator H + lambda is {denom}; raise lam above 0 "
                "when a child has zero Hessian."
            )

    left = GL**2 / (HL + lam)
    right = GR**2 / (HR + lam)
    parent = (GL + GR) ** 2 / (HL + HR + lam)
    gain = 0.5 * (left + right - parent) - gamma

    return RichResult(
        title="XGBoost split gain",
        summary_lines=[("Gain", float(gain)), ("lambda", lam), ("gamma", gamma)],
        payload={
            "gain": float(gain),
            "left_score": float(left),
            "right_score": float(right),
            "parent_score": float(parent),
            "left_weight": float(-GL / (HL + lam)),
            "right_weight": float(-GR / (HR + lam)),
            "should_split": bool(gain > 0),
            "estimate": float(gain),
            "n": 2,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grxgbg: Gain = 0.5[GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l)] - gamma; negative gain = prune"
