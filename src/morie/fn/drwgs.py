# morie.fn -- slice s03 (rootcoder007/morie)
"""Propensity-only, outcome-only, and doubly robust DiD side by side.

Source consulted (FETCHED): Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122 (arXiv:1812.01723).  The whole
argument of the paper is a comparison of three weighting strategies for
the same ATT, so this function computes all three:

  outcome regression  Heckman, Ichimura and Todd (1997), *Review of
                      Economic Studies* 64(4), 605-654:
                      tau^(reg) = E[ (D / E[D]) (dY - mu_0(X)) ]
  inverse propensity  Abadie (2005), *Review of Economic Studies* 72(1),
                      1-19:
                      tau^(ipw) = E[ (D - pi(X)) / ( E[D] (1 - pi(X)) ) dY ]
  doubly robust       Sant'Anna and Zhao (2020) eq. (2.6):
                      tau^(dr) = E[ (w_1 - w_0)(dY - mu_0(X)) ]

The point of the comparison is that the first is consistent only if
mu_0 is right, the second only if pi is right, and the third if *either*
is -- so a large gap between the first two is exactly the diagnostic the
paper is built around.  The 1997 and 2005 papers are paywalled; both
estimands are quoted in their standard published form, and both are
restated in section 2 of Sant'Anna and Zhao, which was fetched.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dr_weighting_strategy"]


def dr_weighting_strategy(y, D, X=None, y0=None):
    """Compute the three ATT estimators and their spread.

    Returns
    -------
    RichResult with payload:
        estimate : the DR estimate
        tau_reg, tau_ipw, tau_dr
        spread   : max - min over the three
        se       : influence-function SE of the DR estimate
    """
    dy = k.vec(y)
    if y0 is not None:
        y00 = k.vec(y0)
        dy = [dy[i] - y00[i] for i in range(len(dy))]
    d = k.vec(D)
    n = len(dy)
    fit = k.drdid_panel(dy, d, X)
    pi = fit["pi"]
    mu0 = fit["mu0"]
    ed = 0.0
    for x in d:
        ed += x / n
    treg = 0.0
    for i in range(n):
        treg += (d[i] / (n * ed)) * (dy[i] - mu0[i]) if ed > 0.0 else 0.0
    tipw = 0.0
    for i in range(n):
        if ed > 0.0:
            tipw += ((d[i] - pi[i]) / (ed * (1.0 - pi[i]))) * dy[i] / n
    tdr = fit["tau"]
    vals = [treg, tipw, tdr]
    return RichResult(
        title="DiD weighting strategies",
        summary_lines=[("DR", tdr), ("IPW", tipw), ("REG", treg)],
        payload={
            "estimate": tdr,
            "tau_reg": treg,
            "tau_ipw": tipw,
            "tau_dr": tdr,
            "spread": max(vals) - min(vals),
            "se": fit["se"],
            "n": n,
            "method": "Outcome-regression (Heckman et al. 1997), IPW (Abadie 2005) and DR (Sant'Anna and Zhao 2020) ATT",
        },
    )


def cheatsheet():
    return "drwgs: DR weighting strategy comparison"
