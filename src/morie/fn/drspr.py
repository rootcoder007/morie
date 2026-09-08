# morie.fn -- function file (rootcoder007/morie)
"""Direct and spillover ATT under an exposure mapping.

Aronow, P. M. and Samii, C. (2017), *Annals of Applied Statistics*
11(4), 1912-1947, doi:10.1214/16-AOAS1005, replace the binary treatment
with an exposure mapping: each unit falls into one of several exposure
conditions determined by its own assignment and by its neighbours', and
a causal contrast is defined between two conditions rather than between
treated and control.

With a binary own-treatment D and a binary neighbourhood exposure E the
three conditions used here are

    direct    D = 1
    spillover D = 0, E = 1
    control   D = 0, E = 0   (the reference condition)

and each contrast against the reference is estimated by the doubly
robust moment of Sant'Anna and Zhao (2020), eq. (2.6), restricted to the
two conditions being compared.  Because the reference condition is
shared, the two contrasts are on the same scale and their sum is the
total effect on a directly-treated, exposed unit under additivity.

The stakes of separating them are the paper's own point: pooling
spillover units into the control group biases the direct effect by
exactly the spillover effect times the exposed share of the controls,
which is the degenerate identity this module can be checked against.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_spillover"]


def dr_spillover(y, D, X=None, exposure=None):
    """DR direct effect and DR spillover effect against a common control.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Own binary treatment.
    X : 2-D array-like, optional
        Baseline covariates.
    exposure : array-like, optional
        Neighbourhood exposure; any strictly positive value counts as
        exposed.  ``None`` means nobody is exposed and the spillover
        contrast is not identified.

    Returns
    -------
    result : dict
        Keys: estimate (direct ATT), att_direct, att_spillover,
        se_direct, se_spillover, total, n_direct, n_spill, n_control, n.

    References
    ----------
    Aronow & Samii (2017), Ann. Appl. Stat. 11(4):1912-1947,
    doi:10.1214/16-AOAS1005.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    ex = k.vec(exposure) if exposure is not None else [0.0] * n
    if len(ex) != n:
        raise ValueError("exposure must have the same length as y")
    Xr = k.mat(X) if X is not None else None
    e = [1.0 if v > 0.0 else 0.0 for v in ex]
    ctrl = [i for i in range(n) if dv[i] < 0.5 and e[i] < 0.5]
    dirt = [i for i in range(n) if dv[i] >= 0.5]
    spil = [i for i in range(n) if dv[i] < 0.5 and e[i] >= 0.5]

    def _fit(a, b):
        idx = a + b
        if len(a) == 0 or len(b) == 0 or len(idx) < 3:
            return float("nan"), float("nan")
        lab = [1.0] * len(a) + [0.0] * len(b)
        f = k.drdid_panel([yv[i] for i in idx], lab,
                          [Xr[i] for i in idx] if Xr is not None else None)
        return f["tau"], f["se"]

    td, sd = _fit(dirt, ctrl)
    ts, ss = _fit(spil, ctrl)
    tot = td + ts if (td == td and ts == ts) else float("nan")
    return RichResult(
        title="DR-DiD with spillover",
        summary_lines=[("exposure conditions", 3)],
        payload={
            "estimate": td,
            "att_direct": td,
            "att_spillover": ts,
            "se_direct": sd,
            "se_spillover": ss,
            "total": tot,
            "n_direct": float(len(dirt)),
            "n_spill": float(len(spil)),
            "n_control": float(len(ctrl)),
            "n": n,
            "method": "DR-DiD with spillover",
        },
    )


def cheatsheet():
    return "drspr: DR-DiD with spillover"


# compact alias per ledger/NAMING.md
drspillover = dr_spillover
