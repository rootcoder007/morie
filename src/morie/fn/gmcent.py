# morie.fn -- slice s03 (rootcoder007/morie)
"""Grand-mean centering of a covariate.

Source consulted: Enders, C. K. and Tofighi, D. (2007).  Centering
predictor variables in cross-sectional multilevel models: a new look at
an old issue.  *Psychological Methods* 12(2), 121-138.  Their equation
for centering at the grand mean (CGM) is

    x_ij(CGM) = x_ij - xbar..

where xbar.. is the mean over all observations in all clusters.  The
paper is paywalled; the transformation is arithmetic and is quoted in
its standard published form.  CGM leaves the within-cluster covariance
structure of the predictor intact and, unlike centering within cluster,
does *not* purge the between-cluster component -- which is exactly the
distinction Enders and Tofighi draw.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["grand_mean_centering"]


def grand_mean_centering(y):
    """Centre a covariate at its grand mean.

    Parameters
    ----------
    y : array-like
        The covariate, pooled over all clusters.

    Returns
    -------
    RichResult with payload:
        estimate : the centred vector x - xbar..
        centered : same as estimate
        grand_mean, sd, n
    """
    v = k.vec(y)
    gm = k.mean(v)
    c = [x - gm for x in v]
    return RichResult(
        title="Grand-mean centering (CGM)",
        summary_lines=[("grand mean", gm)],
        payload={
            "estimate": c,
            "centered": c,
            "grand_mean": gm,
            "sd": k.sd(v, 1) if len(v) > 1 else float("nan"),
            "n": len(v),
            "method": "Grand-mean centering of a level-1 or level-2 covariate",
        },
    )


def cheatsheet():
    return "gmcent: Grand-mean centering for level-1 or level-2 covariate"
