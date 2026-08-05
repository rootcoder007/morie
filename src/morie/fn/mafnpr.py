# morie.fn -- function file (rootcoder007/morie)
"""Funnel-plot coordinates with pseudo-confidence contours."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_funnel_plot_data"]


def ma_funnel_plot_data(yi, se_i, level=0.95):
    """The coordinates of a funnel plot, and the funnel itself.

    A funnel plot is only interpretable against its own null: without the
    contour that a no-bias world would produce, an eye reading the scatter
    is reading noise.  The contour is the fixed-effect summary plus and
    minus a critical value times each study's own standard error, which is
    why the plot is a funnel rather than a band.

    Formula: points ``(y_i, se_i)``; contour ``theta_FE +- z_{1-alpha/2}
    se`` with ``theta_FE = sum(y_i/v_i)/sum(1/v_i)`` -- Light & Pillemer
    (1984), Chapter 3.

    Parameters
    ----------
    yi : array-like
        Study effect estimates.
    se_i : array-like
        Their standard errors, strictly positive.
    level : float, default 0.95
        Contour level.

    Returns
    -------
    RichResult
        ``x_funnel`` (the effects), ``y_funnel`` (the standard errors),
        ``precision``, ``center``, ``ci_lo``, ``ci_hi``, ``k``.

    References
    ----------
    Light, R. J. and Pillemer, D. B. (1984).  Summing Up: The Science of
    Reviewing Research.  Harvard University Press, Chapter 3.
    """
    y = [float(t) for t in core.vec(yi)]
    s = [float(t) for t in core.vec(se_i)]
    k = len(y)
    if k == 0:
        raise ValueError("no studies")
    if len(s) != k:
        raise ValueError("effects and standard errors must have equal length")
    if any(t <= 0.0 for t in s):
        raise ValueError("standard errors must be strictly positive")
    w = [1.0 / (s[i] * s[i]) for i in range(k)]
    sw = sum(w)
    center = sum(w[i] * y[i] for i in range(k)) / sw
    z = core.qnorm(1.0 - (1.0 - float(level)) / 2.0)
    return RichResult(payload={
        "x_funnel": y, "y_funnel": s,
        "precision": [1.0 / t for t in s],
        "center": center,
        "ci_lo": [center - z * t for t in s],
        "ci_hi": [center + z * t for t in s],
        "se_center": (1.0 / sw) ** 0.5, "z_crit": z, "k": k,
        "method": "Funnel-plot coordinates with pseudo-confidence contours"})


def cheatsheet():
    return "mafnpr: funnel-plot coordinates and their pseudo-confidence contours"
