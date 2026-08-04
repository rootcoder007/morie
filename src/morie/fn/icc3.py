"""ICC(3,1) two-way mixed-effects intraclass correlation (consistency)."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .icc1 import _balanced
from .icc2 import _mean_squares

__all__ = ["icc_two_way_mixed"]


def icc_two_way_mixed(y, subject, rater):
    """Two-way mixed-effects, single rater, consistency: ICC(3,1).

    Shrout, P. E. and Fleiss, J. L. (1979), "Intraclass correlations:
    uses in assessing rater reliability", *Psychological Bulletin*
    86(2), 420-428, doi:10.1037/0033-2909.86.2.420, is the primary
    source; it is closed access with no open copy in any repository
    (Unpaywall reports oa_status "closed").  The two-way ANOVA is the
    one printed in Hedderich, J., Sachs, L. and Reynarowych, Z.,
    *Applied Statistics: Methods Using R*, Springer, Section 6.16,
    pp. 427-428, "ANOVA according to Shrout-Fleiss"; that book stops at
    types 1 and 2 and does not print type 3, so the type-3 ratio here

        ICC(3,1) = (BMS - EMS) / (BMS + (k - 1) EMS)

    is the type-2 ratio of p. 428 with the rater-variance term
    k (JMS - EMS) / n dropped, which is what treating the k raters as
    fixed rather than sampled does to the denominator.  Two consequences
    are used as checks instead of a printed number, since none was
    available: ICC(3,1) equals ICC(2,1) exactly when JMS = EMS, and
    ICC(3,1) = 1 exactly when the raters differ only by an additive
    constant, where ICC(2,1) is strictly smaller.

    Parameters
    ----------
    y : array-like
        Ratings in long format.
    subject : array-like
        Subject of each rating.
    rater : array-like
        Rater of each rating; the design must be complete and balanced.

    Returns
    -------
    estimate : ICC(3,1)
    bms, wms, jms, ems : the four mean squares
    """
    rows, n, k = _balanced(y, subject, "icc_two_way_mixed")
    rs = core.vec(rater)
    if len(rs) != n * k:
        raise ValueError("icc_two_way_mixed: rater must have one entry per rating")
    lv = []
    for r in rs:
        if r not in lv:
            lv.append(r)
    if len(lv) != k:
        raise ValueError("icc_two_way_mixed: the number of raters must match the ratings per subject")
    ms = _mean_squares(rows, n, k)
    den = ms["bms"] + (k - 1) * ms["ems"]
    if den == 0.0:
        raise ValueError("icc_two_way_mixed: the ratings carry no variance")
    out = dict(ms)
    out.update({
        "estimate": (ms["bms"] - ms["ems"]) / den,
        "n": n,
        "k": k,
        "method": "ICC(3,1) two-way mixed single rater (consistency)",
    })
    return RichResult(payload=out)


def cheatsheet():
    return "icc3: ICC(3,1) two-way mixed single rater (consistency)"
