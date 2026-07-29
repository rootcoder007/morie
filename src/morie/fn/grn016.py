# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logistic regression class prediction at a 50% probability threshold."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ch4_logistic_regression_prediction"]

_METHOD = "Logistic regression thresholded prediction"


def geron_ch4_logistic_regression_prediction(p_hat, threshold=0.5):
    r"""Threshold estimated probabilities into class labels.

    .. math::
        \hat y = \begin{cases}
            0 & \hat p < 0.5\\
            1 & \hat p \ge 0.5
        \end{cases}

    The ``>=`` matters at the boundary: a probability of exactly 0.5 is
    predicted positive, matching Géron.  Since :math:`\hat p \ge 0.5`
    exactly when the logit :math:`\theta^{\top}x \ge 0`, this is the same
    decision as the sign of the score -- the sigmoid only makes it
    readable.

    Parameters
    ----------
    p_hat : array-like
        Estimated probabilities in ``[0, 1]``.
    threshold : float, optional
        Decision threshold in ``[0, 1]``, default 0.5. Raising it trades
        recall for precision.

    Returns
    -------
    RichResult
        Payload keys ``y_hat``, ``positive_rate``, ``margin``
        (:math:`\hat p - \text{threshold}`), ``threshold``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron (2026), Ch 4, Eq 4-16, p. 168.

    Examples
    --------
    >>> r = geron_ch4_logistic_regression_prediction([0.2, 0.5, 0.9])
    >>> r["y_hat"]
    [0, 1, 1]
    >>> round(r["positive_rate"], 10)
    0.6666666667

    Exactly 0.5 goes positive, not negative:

    >>> geron_ch4_logistic_regression_prediction(0.5)["y_hat"]
    1
    """
    p = np.asarray(p_hat, dtype=float)
    if p.size == 0:
        raise ValueError("p_hat is empty.")
    if not np.all(np.isfinite(p)):
        raise ValueError("p_hat contains non-finite values.")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(
            f"p_hat must be probabilities in [0, 1], got range "
            f"[{float(p.min())}, {float(p.max())}]; pass scores through a sigmoid first."
        )
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}.")

    yhat = (p >= threshold).astype(int)
    est = int(yhat) if yhat.ndim == 0 else yhat.tolist()
    return RichResult(
        title="Logistic regression prediction",
        summary_lines=[("Threshold", threshold), ("Positive rate", float(yhat.mean()))],
        payload={
            "y_hat": est,
            "positive_rate": float(yhat.mean()),
            "margin": float(p - threshold) if p.ndim == 0 else (p - threshold).tolist(),
            "threshold": threshold,
            "estimate": est,
            "n": int(p.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn016: y_hat = 1 iff p_hat >= 0.5 (boundary goes positive); threshold= to trade recall for precision"
