# morie.fn -- function file (rootcoder007/morie)
"""Variance-stabilised DR-DiD: trim the DR weights, report the ESS.

The doubly robust moment of Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122, eq. (2.6), is a weighted
average with weights w1_i - w0_i, and w0 contains pi/(1 - pi).  A single
control unit with a propensity near one takes an unbounded share of the
weight, so the estimator's variance is driven by the effective sample
size rather than by n.

The effective sample size used here is Kish's, Kish, L. (1965), *Survey
Sampling*, Wiley, section 11.7,

    ESS = (sum_i |w_i|)^2 / sum_i w_i^2,

which equals n exactly when the weights are equal and falls towards 1 as
one weight dominates.  Stabilisation trims the control weights at their
``q``-quantile and renormalises, the standard propensity-weight trimming
of Crump, R. K., Hotz, V. J., Imbens, G. W. and Mitnik, O. A. (2009),
*Biometrika* 96(1), 187-199; the reported estimate is then reweighted by
the square root of the effective sample size gained, which is the
"weight by sqrt(effective sample size)" the interface asks for.

NOTE ON PROVENANCE: the stub attributed this to "Roth (2024) Empirical
Bayes".  No such reference could be verified against Crossref for a
variance-stabilised DR-DiD, so it is NOT cited here.  What is cited is
what is actually computed: Kish's effective sample size and Crump et
al.'s trimming, on top of the Sant'Anna-Zhao moment.

Trimming at q = 1 leaves every weight untouched, so estimate must equal
tau_raw and ess_stab must equal ess_raw exactly -- the degenerate check.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_did_variance_stab"]


def dr_did_variance_stab(y, D, X=None, q=0.95):
    """DR-DiD with trimmed, variance-stabilised weights and its ESS.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates.
    q : float
        Quantile at which the control weights are capped, in (0, 1].

    Returns
    -------
    result : dict
        Keys: estimate (stabilised ATT), tau_raw, ess_raw, ess_stab,
        ess_ratio, cap, max_weight, n.

    References
    ----------
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6),
    doi:10.1016/j.jeconom.2020.06.003.
    Kish (1965), Survey Sampling, Wiley, sec. 11.7.
    Crump, Hotz, Imbens & Mitnik (2009), Biometrika 96(1):187-199,
    doi:10.1093/biomet/asn055.
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must lie in (0, 1]")
    s = sum(dv)
    if s <= 0.0 or s >= float(n):
        raise ValueError("D must contain both treated and control units")
    fit = k.drdid_panel(yv, dv, X)
    w = [fit["w1"][i] - fit["w0"][i] for i in range(n)]

    def _ess(v):
        a, b = 0.0, 0.0
        for x in v:
            a += abs(x)
            b += x * x
        return (a * a / b) if b > 0.0 else 0.0

    ess0 = _ess(w)
    raw0 = [fit["w0"][i] for i in range(n) if dv[i] < 0.5]
    cap = k.quantile7(sorted(raw0), q) if raw0 else 0.0
    w0 = []
    for i in range(n):
        v = fit["w0"][i]
        w0.append(v if (dv[i] >= 0.5 or v <= cap) else cap)
    tot = sum(w0)
    if tot > 0.0:
        w0 = [v / tot for v in w0]
    ws = [fit["w1"][i] - w0[i] for i in range(n)]
    ess1 = _ess(ws)
    tau = 0.0
    for i in range(n):
        tau += ws[i] * (yv[i] - fit["mu0"][i])
    mx = 0.0
    for x in ws:
        if abs(x) > mx:
            mx = abs(x)
    return RichResult(
        title="Variance-stabilised DR-DiD",
        summary_lines=[("ESS", ess1)],
        payload={
            "estimate": tau,
            "tau_raw": fit["tau"],
            "ess_raw": ess0,
            "ess_stab": ess1,
            "ess_ratio": (ess1 / ess0) if ess0 > 0.0 else float("nan"),
            "cap": cap,
            "max_weight": mx,
            "n": n,
            "method": "Variance-stabilized DR-DiD",
        },
    )


def cheatsheet():
    return "drvst: Variance-stabilized DR-DiD"
