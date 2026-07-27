# morie.fn -- function file (rootcoder007/morie)
"""Standard error of theta from test information (IRT)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["sem_theta"]


def sem_theta(theta, items):
    r"""Standard error of an ability estimate from the test information.

    For 2PL items with discriminations :math:`a_i` and difficulties
    :math:`b_i`,

    .. math:: I(\theta) = \sum_i a_i^2 P_i(\theta)\,(1 - P_i(\theta)),
              \qquad SE(\theta) = 1 / \sqrt{I(\theta)}

    with :math:`P_i(\theta) = 1 / (1 + e^{-a_i(\theta - b_i)})`
    (Lord 1980, Ch. 5). Information peaks where items are matched to
    the ability (:math:`b_i \approx \theta`), so the SE is smallest
    there -- the fact adaptive testing exploits. This replaces a
    placeholder that ran a KS normality test on theta.

    Parameters
    ----------
    theta : float or array-like
        Ability value(s) at which to evaluate.
    items : array-like, shape (k, 2)
        Item parameters as (a, b) rows.

    Returns
    -------
    RichResult
        keys: ``se`` / ``estimate``, ``information``, ``theta``,
        ``n_items``, ``method``.

    References
    ----------
    Lord, F. M. (1980). *Applications of Item Response Theory to
    Practical Testing Problems*. Erlbaum. Ch. 5 (information and the
    standard error of measurement).
    """
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    it = np.asarray(items, dtype=float)
    if it.ndim != 2 or it.shape[1] != 2:
        raise ValueError(f"items must be (k, 2) rows of (a, b), got {it.shape}.")
    if it.shape[0] < 1:
        raise ValueError("Need at least one item.")
    a = it[:, 0][None, :]
    b = it[:, 1][None, :]
    P = 1.0 / (1.0 + np.exp(-a * (th[:, None] - b)))
    info = (a**2 * P * (1 - P)).sum(axis=1)
    if np.any(info <= 0):
        raise ValueError("test information is zero; SE undefined.")
    se = 1.0 / np.sqrt(info)
    scalar = np.isscalar(theta) or np.asarray(theta).ndim == 0
    return RichResult(
        payload={
            "se": float(se[0]) if scalar else se,
            "estimate": float(se[0]) if scalar else se,
            "information": float(info[0]) if scalar else info,
            "theta": float(th[0]) if scalar else th,
            "n_items": int(it.shape[0]),
            "method": "SE(theta) = 1/sqrt(I(theta)), 2PL test information",
        }
    )


def cheatsheet():
    return "semthe: SE of ability from 2PL test information (Lord 1980)"
