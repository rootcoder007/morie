"""ICC(2,1) two-way random-effects intraclass correlation."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .icc1 import _balanced

__all__ = ["icc_two_way_random"]


def _mean_squares(rows, n, k):
    """The Shrout-Fleiss two-way ANOVA of Hedderich et al., pp. 427-428."""
    tot = 0.0
    tot2 = 0.0
    ssa = 0.0
    colsum = [0.0] * k
    for r in rows:
        s = 0.0
        for j in range(k):
            e = r[j]
            s += e
            tot += e
            tot2 += e * e
            colsum[j] += e
        ssa += s * s / k
    corr = tot * tot / (n * k)
    sst = tot2 - corr
    ssa -= corr
    ssb = 0.0
    for c in colsum:
        ssb += c * c / n
    ssb -= corr
    sse = sst - ssa - ssb
    return {
        "bms": ssa / (n - 1),
        "wms": (sst - ssa) / (n * (k - 1)),
        "jms": ssb / (k - 1),
        "ems": sse / ((n - 1) * (k - 1)),
    }


def icc_two_way_random(y, subject, rater):
    """Two-way random-effects, single rater, absolute agreement: ICC(2,1).

    Shrout, P. E. and Fleiss, J. L. (1979), "Intraclass correlations:
    uses in assessing rater reliability", *Psychological Bulletin*
    86(2), 420-428, doi:10.1037/0033-2909.86.2.420, is the primary
    source; it is closed access with no open copy in any repository
    (Unpaywall reports oa_status "closed"), so the arithmetic was read
    from Hedderich, J., Sachs, L. and Reynarowych, Z., *Applied
    Statistics: Methods Using R*, Springer, Section 6.16, pp. 427-428,
    whose R function labelled "ANOVA according to Shrout-Fleiss" gives

        SS_b = sum(column totals^2)/n - T^2/(n k)
        SS_e = SS_t - SS_a - SS_b
        JMS  = SS_b / (k - 1)
        EMS  = SS_e / ((n - 1)(k - 1))
        ICC(type 2) = (BMS - EMS)
                      / (BMS + (k - 1) EMS + k (JMS - EMS) / n)

    Its printed example -- pituitary height by MRI, k = 3 examiners on
    n = 10 patients with intracranial hypotension, p. 427 -- returns
    ICC type 2 = 0.9759, which is the anchor this module is checked
    against.

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
    estimate : ICC(2,1)
    bms, wms, jms, ems : the four mean squares
    """
    rows, n, k = _balanced(y, subject, "icc_two_way_random")
    rs = core.vec(rater)
    if len(rs) != n * k:
        raise ValueError("icc_two_way_random: rater must have one entry per rating")
    lv = []
    for r in rs:
        if r not in lv:
            lv.append(r)
    if len(lv) != k:
        raise ValueError("icc_two_way_random: the number of raters must match the ratings per subject")
    ms = _mean_squares(rows, n, k)
    den = ms["bms"] + (k - 1) * ms["ems"] + k * (ms["jms"] - ms["ems"]) / n
    if den == 0.0:
        raise ValueError("icc_two_way_random: the ratings carry no variance")
    out = dict(ms)
    out.update({
        "estimate": (ms["bms"] - ms["ems"]) / den,
        "n": n,
        "k": k,
        "method": "ICC(2,1) two-way random single rater",
    })
    return RichResult(payload=out)


def cheatsheet():
    return "icc2: ICC(2,1) two-way random single rater"
