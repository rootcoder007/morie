# morie.fn -- function file (rootcoder007/morie)
"""DiD with a continuous treatment: ATT(d|d) levels and ACRT slopes.

Source opened: Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C.
(2024).  Difference-in-differences with a continuous treatment.
arXiv:2107.02637, Section 3 (page 7 of the working paper), which defines
the two families of building-block parameters

    ATT(d|d') = E[Y_{t=2}(d) - Y_{t=2}(0) | D = d']
    ACRT(d_j|d_k) = E[Y(d_j) - Y(d_{j-1}) | D = d_k] / (d_j - d_{j-1})

and is explicit that these are different objects: all the ATT(d|d) can
be large and positive while the ACRT(d|d) are zero or negative, so a
policy that raises everyone's dose need not have the effect an ATT
suggests.  This module therefore returns the level curve and the slope
curve, never one standing in for the other.

Each ATT(d|d) is estimated by the doubly robust panel moment of
Sant'Anna, P. H. C. and Zhao, J. (2020), *Journal of Econometrics*
219(1), 101-122, eq. (2.6), comparing the units at dose d against the
zero-dose units.  The discrete ACRT is the first difference of that
curve divided by the dose spacing, exactly as printed above; with a
single positive dose the design is binary and ACRT is not defined, so
it is returned empty rather than faked.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_continuous_treatment"]


def dr_continuous_treatment(y, D_dose, X=None):
    """ATT(d|d) at each observed dose and the ACRT between consecutive doses.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D_dose : array-like
        Non-negative treatment intensity; zero marks an untreated unit.
    X : 2-D array-like, optional
        Baseline covariates.

    Returns
    -------
    result : dict
        Keys: estimate (dose-share-weighted mean ATT over treated
        doses), doses, att, se, acrt, acrt_dose, n_zero, n.

    References
    ----------
    Callaway, Goodman-Bacon & Sant'Anna (2024), arXiv:2107.02637, Sec. 3.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    """
    yv = k.vec(y)
    dv = k.vec(D_dose)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D_dose must have the same length")
    for v in dv:
        if v < 0.0:
            raise ValueError("D_dose must be non-negative")
    Xr = k.mat(X) if X is not None else None
    zero = [i for i in range(n) if dv[i] == 0.0]
    if not zero:
        raise ValueError("no zero-dose units to serve as comparison group")
    doses = sorted(set(v for v in dv if v > 0.0))
    if not doses:
        raise ValueError("no treated unit: every dose is zero")
    att, se, nd = [], [], []
    for d in doses:
        idx = [i for i in range(n) if dv[i] == d] + zero
        lab = [1.0] * (len(idx) - len(zero)) + [0.0] * len(zero)
        nd.append(float(len(idx) - len(zero)))
        if len(idx) < 3:
            att.append(float("nan"))
            se.append(float("nan"))
            continue
        f = k.drdid_panel([yv[i] for i in idx], lab,
                          [Xr[i] for i in idx] if Xr is not None else None)
        att.append(f["tau"])
        se.append(f["se"])
    acrt, adose = [], []
    for j in range(1, len(doses)):
        gap = doses[j] - doses[j - 1]
        acrt.append((att[j] - att[j - 1]) / gap)
        adose.append(doses[j])
    num, den = 0.0, 0.0
    for j in range(len(doses)):
        if att[j] == att[j]:
            num += nd[j] * att[j]
            den += nd[j]
    return RichResult(
        title="DR-DiD with continuous treatment",
        summary_lines=[("dose levels", len(doses))],
        payload={
            "estimate": (num / den) if den > 0.0 else float("nan"),
            "doses": doses,
            "att": att,
            "se": se,
            "acrt": acrt,
            "acrt_dose": adose,
            "n_zero": float(len(zero)),
            "n": n,
            "method": "DR-DiD with continuous treatment intensity",
        },
    )


def cheatsheet():
    return "drctf: DR-DiD with continuous treatment intensity"
