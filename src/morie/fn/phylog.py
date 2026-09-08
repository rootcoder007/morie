# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root-to-tip regression dating (TempEst, Rambaut et al. 2016)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["phylog", "phylogenetic_dating"]


def phylog(dates, divergence):
    """
    Root-to-tip regression for temporal signal and TMRCA estimation.

    For heterochronous sequences on a rooted phylogeny, let t_i be
    the sampling time of sequence i and d_(r,i) its root-to-tip
    genetic distance. Under a strict molecular clock,

        E[d_(r,i)] = u (t_i - t_r)            (eq. 1),

    where u is the substitution rate and t_r the time of the tree
    root. Ordinary least squares of d on t estimates the rate as the
    slope, and the x-intercept estimates t_r, the time of the most
    recent common ancestor. The correlation and R^2 diagnose
    temporal signal, and residuals flag incongruent sequences.

    Parameters
    ----------
    dates : array-like
        Sampling times t_i (decimal years or any linear time scale).
    divergence : array-like
        Root-to-tip distances d_(r,i) from the rooted tree.

    Returns
    -------
    result : RichResult
        Keys: rate (slope u), intercept, tmrca (x-intercept t_r),
        correlation, r_squared, residuals, n, method.

    References
    ----------
    Rambaut, A., Lam, T. T., Max Carvalho, L. and Pybus, O. G.
    (2016), "Exploring the temporal structure of heterochronous
    sequences using TempEst (formerly Path-O-Gen)", Virus Evolution
    2(1), vew007. Equation (1) and the regression interpretation
    (slope = rate, x-intercept = t_r), Section 2, "Root-to-tip
    regression". Local source:
    library/pdf/fetched-wave3/Rambaut-2016-TempEst-VirusEvolution.pdf.
    """
    t = np.atleast_1d(np.asarray(dates, dtype=float))
    d = np.atleast_1d(np.asarray(divergence, dtype=float))
    n = len(t)
    if len(d) != n:
        raise ValueError("dates and divergence must have equal length")
    if n < 3:
        raise ValueError("need at least 3 tips")
    tbar = float(np.mean(t))
    dbar = float(np.mean(d))
    sxx = sum((float(t[i]) - tbar) ** 2 for i in range(n))
    sxy = sum((float(t[i]) - tbar) * (float(d[i]) - dbar)
              for i in range(n))
    syy = sum((float(d[i]) - dbar) ** 2 for i in range(n))
    if sxx <= 0.0:
        raise ValueError("all sampling dates identical")
    u = sxy / sxx
    a = dbar - u * tbar
    tmrca = -a / u if u != 0.0 else float("nan")
    corr = (sxy / math.sqrt(sxx * syy)) if syy > 0.0 else float("nan")
    resid = np.asarray([float(d[i]) - (a + u * float(t[i]))
                        for i in range(n)])
    return RichResult(payload={
        "rate": u,
        "intercept": a,
        "tmrca": tmrca,
        "correlation": corr,
        "r_squared": corr * corr if corr == corr else float("nan"),
        "residuals": resid,
        "n": n,
        "method": "root-to-tip regression dating (Rambaut et al. 2016)",
    })


phylogenetic_dating = phylog


def cheatsheet():
    return ("phylog(dates, divergence) -> TempEst root-to-tip "
            "regression: rate = slope, TMRCA = x-intercept.")
