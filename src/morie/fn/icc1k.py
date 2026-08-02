# morie.fn -- function file (rootcoder007/morie)
"""ICC(1,k): one-way random, average of k ratings."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["icc_one_way_average"]


def icc_one_way_average(y, cluster, rater=None):
    r"""Shrout and Fleiss (1979) Case 1, average-measure form:

    .. math:: ICC(1,k) = \frac{MS_R - MS_W}{MS_R},

    the reliability of the MEAN of :math:`k` ratings when each
    target is rated by a DIFFERENT randomly chosen set of raters --
    so rater identity is not crossed with target and no rater effect
    is estimable. That design assumption is the whole of Case 1: if
    the same raters rated everyone, this is the wrong coefficient
    and ICC(2,k) or ICC(3,k) is right.

    Because the rater and error variance cannot be separated here,
    :math:`MS_W` pools them, and ICC(1,\*) is therefore the SMALLEST
    of the three cases on the same data -- systematic rater
    differences are charged to error. The single-measure form
    ICC(1,1) is returned alongside; the two are related by
    Spearman-Brown exactly, which the tests check.

    Parameters
    ----------
    y : array-like
        Ratings.
    cluster : array-like
        Target (subject) identifier.
    rater : array-like, optional
        Accepted for signature parity with the two-way cases and
        used only to determine k when supplied; Case 1 does not
        model rater identity, and the output says so.

    Returns
    -------
    RichResult
        keys: ``value`` (ICC(1,k)), ``icc_single``, ``k``, ``n``,
        ``MSR``, ``MSW``, ``case``, ``design_assumption``,
        ``smallest_because``, ``method``.

    References
    ----------
    Shrout, P. E. and Fleiss, J. L. (1979), "Intraclass correlations:
    uses in assessing rater reliability", *Psychological Bulletin*
    86:420-428, Case 1 and Table 4.
    """
    yv = np.asarray(y, dtype=float).ravel()
    g = np.asarray(cluster).ravel()
    if yv.size != g.size:
        raise ValueError(f"y has {yv.size} entries and cluster has {g.size}.")
    groups = np.unique(g)
    n = groups.size
    if n < 2:
        raise ValueError(f"need at least 2 targets, got {n}.")
    sizes = np.array([np.sum(g == v) for v in groups])
    if not np.all(sizes == sizes[0]):
        raise ValueError(
            "ICC(1,k) as defined by Shrout-Fleiss assumes k ratings per "
            f"target; the group sizes here are {sorted(set(sizes.tolist()))}. "
            "An unbalanced one-way design needs a variance-components fit, "
            "not this formula.")
    k = int(sizes[0])
    if k < 2:
        raise ValueError(f"need at least 2 ratings per target, got {k}.")
    grand = float(yv.mean())
    means = np.array([yv[g == v].mean() for v in groups])
    ms_r = float(k * np.sum((means - grand) ** 2) / (n - 1))
    ss_w = float(sum(np.sum((yv[g == v] - m) ** 2)
                     for v, m in zip(groups, means)))
    ms_w = ss_w / (n * (k - 1))
    if ms_r <= 0:
        raise ValueError("between-target mean square is zero; every target "
                         "has the same mean and no reliability is defined.")
    icc_k = (ms_r - ms_w) / ms_r
    icc_1 = (ms_r - ms_w) / (ms_r + (k - 1) * ms_w)
    return RichResult(payload={
        "value": icc_k, "icc_single": icc_1, "k": k, "n": int(n),
        "MSR": ms_r, "MSW": ms_w, "case": "ICC(1,k)",
        "design_assumption": "each target rated by a DIFFERENT randomly "
                             "chosen set of raters; rater identity is not "
                             "crossed with target",
        "smallest_because": "rater and error variance cannot be separated, "
                            "so MSW pools them and systematic rater "
                            "differences are charged to error -- ICC(1,*) "
                            "is the smallest of the three cases",
        "rater_ignored": rater is not None,
        "method": "Shrout-Fleiss (1979) ICC(1,k) = (MSR - MSW)/MSR"})


def cheatsheet():
    return "icc1k: Case 1 assumes DIFFERENT raters per target -- smallest ICC, rater effect charged to error"
