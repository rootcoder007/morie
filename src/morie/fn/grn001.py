# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-feature linear model: life satisfaction from GDP per capita."""

import numpy as np

from ._richresult import RichResult
from .grn002 import geron_ch4_linear_regression_prediction

__all__ = ["geron_ch4_simple_linear_life_satisfaction"]

_METHOD = "Life-satisfaction one-feature linear model (Eq 4-1)"


def geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita):
    r"""Géron Eq 4-1, the one-feature model that opens Chapter 4.

    .. math::
        \text{life\_satisfaction} = \theta_0 + \theta_1
        \times \text{GDP\_per\_capita}

    This is Eq 4-2 with ``n = 1``, so the arithmetic is delegated to
    :func:`morie.fn.grn002.geron_ch4_linear_regression_prediction`
    rather than repeated here.

    Parameters
    ----------
    theta_0, theta_1 : float
        Intercept and slope.
    GDP_per_capita : float or array-like
        One country's GDP per capita, or a vector of them.

    Returns
    -------
    RichResult
        Payload keys ``life_satisfaction``, ``theta_0``, ``theta_1``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-1, p. 136.

    Examples
    --------
    The book's fitted line at a GDP per capita of 20 000:

    >>> r = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 20000.0)
    >>> round(r["life_satisfaction"], 4)
    5.832

    Doubling GDP doubles only the slope term, never the intercept:

    >>> r2 = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 40000.0)
    >>> round(r2["life_satisfaction"] - r["life_satisfaction"], 4)
    0.982
    """
    theta_0 = float(theta_0)
    theta_1 = float(theta_1)
    if not np.isfinite(theta_0) or not np.isfinite(theta_1):
        raise ValueError(f"theta_0 and theta_1 must be finite, got {theta_0}, {theta_1}.")
    g = np.asarray(GDP_per_capita, dtype=float)
    if g.ndim > 1:
        raise ValueError(f"GDP_per_capita must be scalar or 1-D, got ndim {g.ndim}.")
    if not np.all(np.isfinite(g)):
        raise ValueError("GDP_per_capita must be finite.")
    if np.any(g < 0):
        raise ValueError("GDP per capita cannot be negative.")

    scalar = g.ndim == 0
    Xm = g.reshape(-1, 1)
    inner = geron_ch4_linear_regression_prediction([theta_0, theta_1], Xm)
    pred = inner["prediction"]
    value = float(pred[0]) if scalar else [float(v) for v in pred]

    return RichResult(
        title="Life satisfaction ~ GDP per capita",
        summary_lines=[("theta_0", theta_0), ("theta_1", theta_1)],
        payload={
            "life_satisfaction": value,
            "theta_0": theta_0,
            "theta_1": theta_1,
            "estimate": value,
            "n": int(Xm.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn001: life_satisfaction = theta_0 + theta_1 * GDP_per_capita (delegates to grn002)"
