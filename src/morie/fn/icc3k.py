# morie.fn -- function file (rootcoder007/morie)
"""ICC(3,k): two-way mixed, average measure."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["icc_two_way_mixed_avg"]


def icc_two_way_mixed_avg(y, subject, rater):
    r"""Shrout and Fleiss (1979) Case 3, average-measure form:

    .. math:: ICC(3,k) = \frac{MS_R - MS_E}{MS_R}
              = 1 - \frac{MS_E}{MS_R},

    the reliability of the mean of :math:`k` ratings when THESE
    raters are the only ones of interest -- a fixed effect, not a
    sample. No rater term appears, so systematic rater differences
    are not charged against reliability at all.

    That is a consistency coefficient, not an agreement one: two
    raters who disagree by a constant offset on every target get
    ICC(3,\*) near 1 and ICC(2,\*) well below it. Which one a study
    should report follows from whether the ratings will be used
    interchangeably (agreement, Case 2) or only relatively
    (consistency, Case 3), and the output carries both plus the
    measured offset so the choice is informed rather than habitual.

    Parameters
    ----------
    y : array-like
        Ratings.
    subject, rater : array-like
        Target and rater identifiers; a complete crossed design.

    Returns
    -------
    RichResult
        keys: ``value`` (ICC(3,k)), ``icc_single``, ``icc2k``,
        ``max_rater_offset``, ``k``, ``n``, ``MSR``, ``MSC``,
        ``MSE``, ``case``, ``consistency_not_agreement``, ``method``.

    References
    ----------
    Shrout, P. E. and Fleiss, J. L. (1979), *Psychological Bulletin*
    86:420-428, Case 3 and Table 4. McGraw and Wong (1996),
    *Psychological Methods* 1:30-46.
    """
    from ._psycho import anova_two_way

    a = anova_two_way(y, subject, rater)
    n, k = a["n"], a["k"]
    msr, msc, mse = a["MSR"], a["MSC"], a["MSE"]
    if msr <= 0:
        raise ValueError("between-target mean square is zero; every target "
                         "has the same mean and no reliability is defined.")
    icc_k = (msr - mse) / msr
    icc_1 = (msr - mse) / (msr + (k - 1) * mse)
    den2 = msr + (msc - mse) / n
    icc2k = (msr - mse) / den2 if den2 > 0 else np.nan
    col_means = a["matrix"].mean(axis=0)
    return RichResult(payload={
        "value": icc_k, "icc_single": icc_1, "icc2k": icc2k,
        "max_rater_offset": float(col_means.max() - col_means.min()),
        "k": int(k), "n": int(n),
        "MSR": msr, "MSC": msc, "MSE": mse, "case": "ICC(3,k)",
        "design_assumption": "THESE raters are the only ones of interest -- "
                             "a fixed effect, not a sample",
        "consistency_not_agreement": "systematic rater offsets are not "
                                     "charged: two raters differing by a "
                                     "constant on every target score near 1 "
                                     "here and well below it on ICC(2,k). "
                                     "Report Case 3 only when the ratings "
                                     "are used relatively, not "
                                     "interchangeably",
        "method": "Shrout-Fleiss (1979) ICC(3,k), two-way mixed, average measure"})


def cheatsheet():
    return "icc3k: consistency, not agreement -- a constant rater offset costs it nothing"
