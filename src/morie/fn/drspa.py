# morie.fn -- function file (rootcoder007/morie)
"""Doubly robust DiD with a spatial lag of treatment among the controls.

Anselin, L. (2003), Spatial externalities, spatial multipliers, and
spatial econometrics, *International Regional Science Review* 26(2),
153-166, doi:10.1177/0160017602250972, is the reference for the object
added here: with a row-standardised contiguity matrix W the spatial lag
Wz of a variable z is the neighbourhood average

    (W z)_i = sum_j w_ij z_j / sum_j w_ij,

and a policy whose effect leaks across boundaries makes Wz a genuine
regressor, not a nuisance.  The lag of the treatment, WD, is therefore
appended to the covariate block of the doubly robust moment of
Sant'Anna and Zhao (2020), eq. (2.6), so that both the propensity score
and the outcome regression condition on how treated a unit's
neighbourhood is.

Reported alongside is the estimate that omits WD.  Their difference is
the spatial confounding the lag absorbs; a W with no off-diagonal mass
makes WD constant, the design column is collinear with the intercept,
and the two estimates must coincide -- that is the degenerate check.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_spatial_did"]


def dr_spatial_did(y, D, X=None, W_neighbors=None):
    """DR-DiD conditioning on the spatial lag of treatment.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates.
    W_neighbors : 2-D array-like, optional
        Non-negative n x n neighbour weights; row-standardised here.
        ``None`` reduces to the plain DR estimator.

    Returns
    -------
    result : dict
        Keys: estimate (with the spatial lag), tau_nospatial,
        spatial_shift, se, wd_mean, wd_sd, n.

    References
    ----------
    Anselin (2003), Int. Reg. Sci. Rev. 26(2):153-166,
    doi:10.1177/0160017602250972.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    Xr = k.mat(X) if X is not None else None
    if Xr is not None and k.nrow(Xr) != n:
        raise ValueError("X must have one row per unit")
    if W_neighbors is None:
        wd = [0.0] * n
    else:
        W = k.mat(W_neighbors)
        if k.nrow(W) != n or k.ncol(W) != n:
            raise ValueError("W_neighbors must be n x n")
        wd = []
        for i in range(n):
            s, num = 0.0, 0.0
            for j in range(n):
                if W[i][j] < 0.0:
                    raise ValueError("W_neighbors must be non-negative")
                s += W[i][j]
                num += W[i][j] * dv[j]
            wd.append(num / s if s > 0.0 else 0.0)
    base = k.drdid_panel(yv, dv, Xr)
    cols = []
    for i in range(n):
        row = list(Xr[i]) if Xr is not None else []
        row.append(wd[i])
        cols.append(row)
    m = k.mean(wd)
    sdv = k.sd(wd) if n > 1 else 0.0
    if sdv <= 1e-12:
        fit = base
    else:
        fit = k.drdid_panel(yv, dv, cols)
    return RichResult(
        title="Spatial DR-DiD",
        summary_lines=[("mean neighbour treatment", m)],
        payload={
            "estimate": fit["tau"],
            "tau_nospatial": base["tau"],
            "spatial_shift": fit["tau"] - base["tau"],
            "se": fit["se"],
            "wd_mean": m,
            "wd_sd": sdv,
            "n": n,
            "method": "Spatial DR-DiD with neighbor effects",
        },
    )


def cheatsheet():
    return "drspa: Spatial DR-DiD with neighbor effects"


# compact alias per ledger/NAMING.md
drspatialdid = dr_spatial_did
