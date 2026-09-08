# morie.fn -- slice s04 (rootcoder007/morie)
"""Reliability: squared predictive ability divided by heritability.

Sources consulted: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, Chapter 4, equation (4.2), which defines the
predictive ability r as Pearson's correlation between observed and
predicted values on the testing set; and Dekkers, J. C. M. (2007),
Prediction of response to marker-assisted and genomic selection using
selection index theory, *Journal of Animal Breeding and Genetics*
124(6), 331-341, which gives the accuracy of prediction of the breeding
value as r_g = r / h, so that the reliability -- the squared accuracy --
is

    rel = r^2 / h^2.

Chapter 4 defines r but never divides it by h; that step is Dekkers's.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["reliability_metric"]


def reliability_metric(r, h2):
    """Reliability of genomic prediction from predictive ability and h^2.

    Parameters
    ----------
    r : array-like
        Predictive ability (correlation of observed with predicted), in
        [-1, 1].
    h2 : array-like
        Narrow-sense heritability, in (0, 1]; recycled against r.

    Returns
    -------
    estimate    : the first reliability
    reliability : the vector r^2 / h^2
    accuracy    : the vector r / sqrt(h^2), Dekkers's r_g
    """
    rr = k.vec(r)
    hh = k.vec(h2)
    if not rr or not hh:
        raise ValueError("reliability_metric: both r and h2 are required")
    if len(rr) != len(hh) and len(rr) != 1 and len(hh) != 1:
        raise ValueError("reliability_metric: r and h2 have incompatible lengths")
    n = max(len(rr), len(hh))
    rel = []
    acc = []
    for i in range(n):
        a = rr[i % len(rr)]
        b = hh[i % len(hh)]
        if a < -1.0 or a > 1.0:
            raise ValueError("reliability_metric: r must lie in [-1, 1]")
        if b <= 0.0 or b > 1.0:
            raise ValueError("reliability_metric: h2 must lie in (0, 1]")
        rel.append(a * a / b)
        acc.append(a / (b ** 0.5))
    return RichResult(
        title="Reliability of genomic prediction",
        summary_lines=[("pairs", n)],
        payload={
            "estimate": rel[0],
            "reliability": rel,
            "accuracy": acc,
            "n": n,
            "method": "rel = r^2 / h^2, with r the Chapter 4 eq. (4.2) predictive ability (Dekkers 2007)",
        },
    )


def cheatsheet():
    return "relbt: Reliability: squared predictive ability divided by heritability"
