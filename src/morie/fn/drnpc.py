# morie.fn -- function file (rootcoder007/morie)
"""Negative-control falsification for the doubly robust DiD estimator.

Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010), Negative
controls: a tool for detecting confounding and bias in observational
studies, *Epidemiology* 21(3), 383-388,
doi:10.1097/EDE.0b013e3181d61eeb, define a negative control outcome as
one that cannot plausibly be affected by the exposure but shares its
confounding structure.  Its estimated effect is therefore an estimate of
the bias: under no unmeasured confounding it is zero, and a
significantly non-zero value falsifies the design.

Both outcomes are run through the same doubly robust moment of
Sant'Anna and Zhao (2020), eq. (2.6), so the two estimates differ only
in the outcome, and

    decision  reject  <=>  |tau_neg| > z_{1-alpha/2} se_neg
    tau_adj   = tau_main - tau_neg

is the difference-in-differences-in-differences that subtracts the
estimated bias.  The decision is a hypothesis test, so it is reported as
a decision and not folded into the point estimate: a design that passes
the falsification is not thereby validated, it is merely not refuted.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_did_neg_control"]


def dr_did_neg_control(y_main, y_neg, D, X=None, alpha=0.05):
    """DR-DiD on a main outcome plus a negative-control falsification.

    Parameters
    ----------
    y_main : array-like
        Outcome change for the outcome of interest.
    y_neg : array-like
        Outcome change for the negative control outcome.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates.
    alpha : float
        Two-sided level of the falsification test, in (0, 1).

    Returns
    -------
    result : dict
        Keys: estimate (main ATT), tau_main, tau_neg, se_main, se_neg,
        z_neg, crit, falsified (1 = design refuted), tau_adj, n.

    References
    ----------
    Lipsitch, Tchetgen Tchetgen & Cohen (2010), Epidemiology
    21(3):383-388, doi:10.1097/EDE.0b013e3181d61eeb.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    """
    ym = k.vec(y_main)
    yn = k.vec(y_neg)
    dv = k.vec(D)
    n = len(ym)
    if n == 0:
        raise ValueError("empty input: y_main has no observations")
    if len(yn) != n or len(dv) != n:
        raise ValueError("y_main, y_neg and D must have the same length")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    s = sum(dv)
    if s <= 0.0 or s >= float(n):
        raise ValueError("D must contain both treated and control units")
    Xr = k.mat(X) if X is not None else None
    fm = k.drdid_panel(ym, dv, Xr)
    fn = k.drdid_panel(yn, dv, Xr)
    crit = k.qnorm(1.0 - alpha / 2.0)
    z = fn["tau"] / fn["se"] if fn["se"] > 0.0 else float("nan")
    bad = 1.0 if (z == z and abs(z) > crit) else 0.0
    return RichResult(
        title="DR-DiD with a negative control outcome",
        summary_lines=[("falsified", bad)],
        payload={
            "estimate": fm["tau"],
            "tau_main": fm["tau"],
            "tau_neg": fn["tau"],
            "se_main": fm["se"],
            "se_neg": fn["se"],
            "z_neg": z,
            "crit": crit,
            "falsified": bad,
            "tau_adj": fm["tau"] - fn["tau"],
            "n": n,
            "method": "DR-DiD with negative control outcome",
        },
    )


def cheatsheet():
    return "drnpc: DR-DiD with negative control outcome"
