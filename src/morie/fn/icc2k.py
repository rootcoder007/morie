# morie.fn -- function file (rootcoder007/morie)
"""ICC(2,k): two-way random, average measure."""

import numpy as np

from ._richresult import RichResult

__all__ = ["icc_two_way_random_avg"]


def icc_two_way_random_avg(y, subject, rater):
    r"""Shrout and Fleiss (1979) Case 2, average-measure form:

    .. math:: ICC(2,k) = \frac{MS_R - MS_E}
              {MS_R + (MS_C - MS_E)/n},

    the reliability of the mean of :math:`k` ratings when the raters
    are a RANDOM SAMPLE from a larger population and the result is
    meant to generalise to other raters.

    The :math:`(MS_C - MS_E)/n` term is the difference from Case 3
    and the whole point: it charges SYSTEMATIC RATER DIFFERENCES
    against reliability, because a future rater drawn from the same
    population would bring their own bias. ICC(3,k) drops that term
    -- treating these raters as the only ones of interest -- and is
    therefore always at least as large. Reporting ICC(3,k) when the
    raters were a sample is the standard way to overstate
    reliability, and the two are returned together here so the gap
    is visible.

    Parameters
    ----------
    y : array-like
        Ratings.
    subject, rater : array-like
        Target and rater identifiers; a complete crossed design.

    Returns
    -------
    RichResult
        keys: ``value`` (ICC(2,k)), ``icc_single``, ``icc3k``
        (the fixed-rater counterpart), ``rater_penalty``, ``k``,
        ``n``, ``MSR``, ``MSC``, ``MSE``, ``case``,
        ``design_assumption``, ``method``.

    References
    ----------
    Shrout, P. E. and Fleiss, J. L. (1979), *Psychological Bulletin*
    86:420-428, Case 2 and Table 4. McGraw and Wong (1996),
    *Psychological Methods* 1:30-46, for the absolute-agreement
    framing.
    """
    from ._psycho import anova_two_way

    a = anova_two_way(y, subject, rater)
    n, k = a["n"], a["k"]
    msr, msc, mse = a["MSR"], a["MSC"], a["MSE"]
    den_k = msr + (msc - mse) / n
    if den_k <= 0:
        raise ValueError("the ICC(2,k) denominator is not positive; the "
                         "variance components do not admit a reliability "
                         "on this table.")
    icc_k = (msr - mse) / den_k
    den_1 = msr + (k - 1) * mse + k * (msc - mse) / n
    icc_1 = (msr - mse) / den_1 if den_1 > 0 else np.nan
    icc3k = (msr - mse) / msr if msr > 0 else np.nan
    return RichResult(payload={
        "value": icc_k, "icc_single": icc_1, "icc3k": icc3k,
        "rater_penalty": float(icc3k - icc_k),
        "k": int(k), "n": int(n),
        "MSR": msr, "MSC": msc, "MSE": mse, "case": "ICC(2,k)",
        "design_assumption": "raters are a RANDOM SAMPLE from a larger "
                             "population and the result should generalise "
                             "to other raters",
        "why_smaller_than_icc3": "the (MSC - MSE)/n term charges systematic "
                                 "rater differences against reliability, "
                                 "because a future rater brings their own "
                                 "bias; ICC(3,k) drops it and is always at "
                                 "least as large",
        "method": "Shrout-Fleiss (1979) ICC(2,k), two-way random, average measure"})


def cheatsheet():
    return "icc2k: random raters pay the (MSC - MSE)/n penalty -- ICC(3,k) does not, and is always bigger"
