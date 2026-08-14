"""ICC(1) one-way random-effects intraclass correlation."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["icc_one_way"]


def _balanced(y, group, who):
    """Long-format (value, group) into a subjects-by-raters matrix."""
    ys = core.vec(y)
    gs = core.vec(group)
    if len(ys) == 0:
        raise ValueError(who + ": y is empty")
    if len(gs) != len(ys):
        raise ValueError(who + ": y and the grouping must have the same length")
    levels = []
    for g in gs:
        if g not in levels:
            levels.append(g)
    levels.sort()
    rows = [[] for _ in levels]
    for i in range(len(ys)):
        rows[levels.index(gs[i])].append(ys[i])
    k = len(rows[0])
    for r in rows:
        if len(r) != k:
            raise ValueError(who + ": the design must be balanced -- every subject needs the same number of ratings")
    if len(rows) < 2:
        raise ValueError(who + ": need at least two subjects")
    if k < 2:
        raise ValueError(who + ": need at least two ratings per subject")
    return rows, len(rows), k


def icc_one_way(y, cluster):
    """One-way random-effects ICC(1,1).

    Shrout, P. E. and Fleiss, J. L. (1979), "Intraclass correlations:
    uses in assessing rater reliability", *Psychological Bulletin*
    86(2), 420-428, doi:10.1037/0033-2909.86.2.420, is the primary
    source for the three forms.  That paper is closed access with no
    open copy in any repository (checked against Unpaywall, which
    reports oa_status "closed" and an empty oa_locations list), so the
    arithmetic below was read from a source that reproduces it:
    Hedderich, J., Sachs, L. and Reynarowych, Z., *Applied Statistics:
    Methods Using R*, Springer, Section 6.16 "Agreement and Precision
    of Measurements", pp. 427-428, whose worked R function is labelled
    "ANOVA according to Shrout-Fleiss" and gives

        SS_t = sum x^2 - T^2/(n k)
        SS_a = sum(row totals^2)/k - T^2/(n k)
        BMS  = SS_a / (n - 1)
        WMS  = (SS_t - SS_a) / (n (k - 1))
        ICC(type 1) = (BMS - WMS) / (BMS + (k - 1) WMS)

    with T the grand total, n subjects and k ratings each.

    Parameters
    ----------
    y : array-like
        Ratings in long format.
    cluster : array-like
        Subject each rating belongs to; the design must be balanced.

    Returns
    -------
    estimate : ICC(1,1)
    bms, wms : the between- and within-subject mean squares
    """
    rows, n, k = _balanced(y, cluster, "icc_one_way")
    tot = 0.0
    tot2 = 0.0
    ssa = 0.0
    for r in rows:
        s = 0.0
        for e in r:
            s += e
            tot += e
            tot2 += e * e
        ssa += s * s / k
    corr = tot * tot / (n * k)
    sst = tot2 - corr
    ssa -= corr
    bms = ssa / (n - 1)
    wms = (sst - ssa) / (n * (k - 1))
    den = bms + (k - 1) * wms
    if den == 0.0:
        raise ValueError("icc_one_way: the ratings carry no variance")
    return RichResult(payload={
        "estimate": (bms - wms) / den,
        "bms": bms,
        "wms": wms,
        "sst": sst,
        "ssa": ssa,
        "n": n,
        "k": k,
        "method": "ICC(1) one-way random-effects model",
    })


def cheatsheet():
    return "icc1: ICC(1) one-way random-effects model"

# public names resolved by fn/_lazy_map.json
icconeway = icc_one_way
