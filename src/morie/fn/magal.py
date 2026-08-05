# morie.fn -- function file (rootcoder007/morie)
"""Galbraith (radial) plot coordinates."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_galbraith"]


def ma_galbraith(yi, se_i):
    """Put every study on the same vertical scale before comparing them.

    A forest plot draws each study at its own precision, so the eye
    compares intervals of different widths and cannot see heterogeneity.
    Dividing by the standard error puts every point on a unit-variance
    axis: under homogeneity the points scatter with unit standard
    deviation about a line through the origin, so a study more than two
    units off the line is visibly discrepant regardless of its size.

    Formula: ``z_i = y_i/se_i`` against ``x_i = 1/se_i``; the
    through-the-origin slope ``sum(z_i x_i)/sum(x_i^2)`` is exactly the
    inverse-variance pooled estimate -- Galbraith (1988).

    Parameters
    ----------
    yi : array-like
        Study effect estimates.
    se_i : array-like
        Their standard errors, strictly positive.

    Returns
    -------
    RichResult
        ``z``, ``x``, ``slope``, ``resid`` (the vertical distances from
        the fitted line), ``n_outside_2`` (points beyond two units),
        ``k``.

    References
    ----------
    Galbraith, R. F. (1988).  A note on graphical presentation of
    estimated odds ratios from several clinical trials.  Statistics in
    Medicine 7(8):889-894.  doi:10.1002/sim.4780070807.
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
    z = [y[i] / s[i] for i in range(k)]
    x = [1.0 / s[i] for i in range(k)]
    slope = sum(z[i] * x[i] for i in range(k)) / sum(t * t for t in x)
    resid = [z[i] - slope * x[i] for i in range(k)]
    return RichResult(payload={
        "z": z, "x": x, "slope": slope, "resid": resid,
        "n_outside_2": sum(1 for t in resid if abs(t) > 2.0), "k": k,
        "method": "Galbraith (radial) plot"})


def cheatsheet():
    return "magal: Galbraith radial-plot coordinates and their fitted slope"


# compact alias per ledger/NAMING.md
magalbraith = ma_galbraith
