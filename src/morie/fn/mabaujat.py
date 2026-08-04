# morie.fn -- function file (rootcoder007/morie)
"""Baujat plot coordinates for a meta-analysis."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['baujat', 'ma_baujat_plot_data']


def baujat(yi, vi):
    """Baujat plot coordinates for a meta-analysis.

    The plot separates two things a single influence measure confuses: how much a study inflates heterogeneity, and how much it moves the pooled estimate. A study high on both axes is the one worth investigating; a study high only on the first is merely an outlier that does not matter. Leave-one-out estimates are exact, not approximated, since inverse-variance weighting makes the deletion a subtraction.


    Formula: x_i = w_i (y_i - theta_FE)^2; y_i = (theta_FE - theta_FE(-i))^2 / var(theta_FE(-i))

    Parameters
    ----------
    yi : array-like
        Effect estimates.
    vi : array-like
        Their sampling variances.

    Returns
    -------
    RichResult
        ``x`` (contribution to Q), ``y`` (influence on the pooled estimate), ``theta_fe``, ``theta_loo``, ``k``.

    References
    ----------
    Baujat, Mahe, Pignon and Hill (2002), A graphical method for
    exploring heterogeneity in meta-analyses, Statistics in Medicine
    21:2641-2652.  Paywalled; the two axes are as documented by
    metafor::baujat, the reference implementation -- the x-axis is each
    study's contribution to the Q statistic and the y-axis the
    standardised squared difference between the overall estimate with
    and without that study.
    """
    y = C.vec(yi); v = C.vec(vi)
    k = len(y)
    if any(t <= 0 for t in v):
        raise ValueError("variances must be positive")
    w = [1.0 / t for t in v]
    sw = sum(w)
    th = sum(w[i] * y[i] for i in range(k)) / sw
    xs, ys, loo = [], [], []
    for i in range(k):
        xs.append(w[i] * (y[i] - th) ** 2)
        sw_i = sw - w[i]
        if sw_i <= 0:
            loo.append(float("nan")); ys.append(float("nan")); continue
        th_i = (sum(w[j] * y[j] for j in range(k)) - w[i] * y[i]) / sw_i
        loo.append(th_i)
        ys.append((th - th_i) ** 2 * sw_i)
    return RichResult(payload={
        "x": xs, "y": ys, "theta_fe": th, "theta_loo": loo, "k": k,
        "method": "Baujat plot coordinates"})


ma_baujat_plot_data = baujat


def cheatsheet():
    return "mabaujat: Baujat plot coordinates for a meta-analysis."
