# morie.fn -- slice s03 (rootcoder007/morie)
"""Doubly robust ATT in a changeover (cross-over) design.

Sources consulted: Jones, B. and Kenward, M. G. (2014).  *Design and
Analysis of Cross-Over Trials*, 3rd ed., Chapman and Hall, chapter 2,
for the design itself -- in a two-period, two-sequence cross-over each
unit is observed under both conditions, so the within-unit contrast

    d_i = (Y_(i,treated period) - Y_(i,control period))

removes every time-invariant unit effect, and the sequence-averaged
contrast

    tau = ( dbar_(AB) - dbar_(BA) ) / 2

removes any common period effect.  The book was not available here as a
full text; the two expressions are the standard published form of the
AB/BA analysis and are not in dispute.  Covariate adjustment then uses
the doubly robust moment of Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122 (arXiv:1812.01723 -- FETCHED),
equation (2.6), with the sequence indicator taking the role of D and the
within-unit contrast the role of dY.

Carryover is *not* assumed away silently: the difference between the two
sequence means, which is the carryover contrast under the standard
model, is returned as ``carryover`` so the user can see it.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["changeover_dr"]


def changeover_dr(y, D, period=None, unit=None, X=None):
    """ATT from a two-period cross-over, with a DR covariate adjustment.

    Parameters
    ----------
    y : array-like
        Outcome, long format.
    D : array-like
        Treatment indicator for that unit-period.
    period : array-like
        Period identifier (two levels).
    unit : array-like
        Unit identifier.
    X : 2-D array-like, optional
        Baseline covariates, one row per unit (first row seen wins).

    Returns
    -------
    RichResult with payload:
        estimate  : the DR sequence-averaged ATT
        tau_naive : the unadjusted (dbar_AB - dbar_BA)/2
        carryover : difference of sequence means
        n_units
    """
    yv = k.vec(y)
    d = k.vec(D)
    p = [float(x) for x in period]
    u = [str(x) for x in unit]
    Xr = k.mat(X) if X is not None else None
    pers = sorted(set(p))
    units = []
    for x in u:
        if x not in units:
            units.append(x)
    contrast = []
    seq = []
    xs = []
    for uu in units:
        idx = [i for i in range(len(yv)) if u[i] == uu]
        if len(idx) < 2:
            continue
        i0 = None
        i1 = None
        for i in idx:
            if p[i] == pers[0]:
                i0 = i
            elif p[i] == pers[1]:
                i1 = i
        if i0 is None or i1 is None:
            continue
        # within-unit contrast, oriented treated-minus-control
        if d[i0] > d[i1]:
            contrast.append(yv[i0] - yv[i1])
            seq.append(1.0)
        else:
            contrast.append(yv[i1] - yv[i0])
            seq.append(0.0)
        if Xr is not None:
            xs.append(Xr[i0])
    m1 = [contrast[i] for i in range(len(seq)) if seq[i] > 0.5]
    m0 = [contrast[i] for i in range(len(seq)) if seq[i] < 0.5]
    naive = 0.5 * ((k.mean(m1) if m1 else 0.0) + (k.mean(m0) if m0 else 0.0))
    carry = (k.mean(m1) if m1 else float("nan")) - (k.mean(m0) if m0 else float("nan"))
    if m1 and m0 and len(contrast) >= 3:
        fit = k.drdid_panel(contrast, seq, xs if Xr is not None else None)
        est = fit["tau"]
        se = fit["se"]
    else:
        est = naive
        se = float("nan")
    return RichResult(
        title="Cross-over DR ATT",
        summary_lines=[("ATT", est), ("units", len(contrast))],
        payload={
            "estimate": est,
            "tau_naive": naive,
            "carryover": carry,
            "se": se,
            "n_units": len(contrast),
            "method": "Two-period cross-over ATT with a doubly robust covariate adjustment",
        },
    )


def cheatsheet():
    return "chtchg: DR for changeover designs"


changeoverdr = changeover_dr
