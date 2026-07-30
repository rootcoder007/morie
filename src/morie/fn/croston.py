# morie.fn -- function file (rootcoder007/morie)
"""Croston's method for intermittent demand."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["croston"]


def croston(y, alpha=0.1, variant="croston"):
    r"""Forecast intermittent demand by smoothing size and interval separately.

    Ordinary exponential smoothing fails on demand that is mostly zero: it
    decays toward zero between orders and then jumps, so the forecast is
    wrong almost everywhere. Croston smooths two series instead -- the nonzero
    demand sizes :math:`z` and the intervals between them :math:`p` -- and
    forecasts the rate

    .. math::
        \hat y = \frac{\hat z}{\hat p}.

    The estimator is **biased upward**, by a factor of about
    :math:`1/(1-\alpha/2)`, because the ratio of two smoothed quantities is
    not the smoothed ratio. That is not a subtlety to gloss over: at
    :math:`\alpha = 0.1` it is roughly 5%, and it compounds across an
    inventory. ``variant="sba"`` applies the Syntetos-Boylan correction, which
    is the usual recommendation.

    Parameters
    ----------
    y : array-like
        Demand series, non-negative, typically mostly zero.
    alpha : float
        Smoothing parameter in (0, 1].
    variant : {"croston", "sba"}
        ``"sba"`` applies the bias correction.

    Returns
    -------
    RichResult
        ``forecast``, ``demand_size``, ``interval``, ``bias_factor``,
        ``n_nonzero``, ``intermittency``.

    References
    ----------
    Croston, J. D. (1972). Forecasting and stock control for intermittent
        demands. *Operational Research Quarterly*, 23(3), 289-303.
    Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent
        demand estimates. *International Journal of Forecasting*, 21(2),
        303-314.

    Examples
    --------
    Demand of 10 units every 4 periods gives a rate of 2.5 per period.

    >>> import numpy as np
    >>> y = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]
    >>> r = croston(y, alpha=0.1)
    >>> bool(abs(r["forecast"] - 2.5) < 0.2)
    True

    The plain estimator is biased upward; SBA corrects it downward.

    >>> sba = croston(y, alpha=0.1, variant="sba")
    >>> bool(sba["forecast"] < r["forecast"])
    True
    >>> float(round(sba["bias_factor"], 3))
    0.95

    Intermittency is reported, since Croston is only appropriate when it is
    high.

    >>> bool(r["intermittency"] > 0.7)
    True

    >>> croston([0, 0, 0], alpha=0.1)
    Traceback (most recent call last):
        ...
    ValueError: the series has no nonzero demand
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if np.any(y < 0):
        raise ValueError("demand must be non-negative")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if variant not in ("croston", "sba"):
        raise ValueError('variant must be "croston" or "sba"')
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        raise ValueError("the series has no nonzero demand")

    z = float(y[nz[0]])
    p = float(nz[0] + 1)
    last = nz[0]
    for i in nz[1:]:
        z = alpha * y[i] + (1 - alpha) * z
        p = alpha * (i - last) + (1 - alpha) * p
        last = i
    rate = z / max(p, 1e-12)
    bias = 1.0 - alpha / 2.0 if variant == "sba" else 1.0
    return RichResult(
        title=f"Croston ({variant})",
        summary_lines=[("n", int(y.size)), ("nonzero", int(nz.size)),
                       ("forecast", float(rate * bias))],
        warnings=(["the plain Croston estimator is biased upward by about "
                   "1/(1 - alpha/2); consider variant='sba'"]
                  if variant == "croston" else []),
        payload={
            "forecast": float(rate * bias), "rate": float(rate),
            "demand_size": z, "interval": p, "bias_factor": float(bias),
            "n_nonzero": int(nz.size),
            "intermittency": float(1.0 - nz.size / y.size),
            "alpha": alpha, "variant": variant, "method": "croston",
        },
    )


def cheatsheet():
    return "croston: smooths SIZE and INTERVAL separately; plain version biased UP ~5% at alpha=0.1, use sba"
