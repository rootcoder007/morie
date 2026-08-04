# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic relative efficiency of two estimators."""

import math

from ._richresult import RichResult

__all__ = ["areratio", "asymptotic_relative_efficiency"]


def areratio(var1, var2, n1=1.0, n2=1.0):
    """Ratio of asymptotic variances, on a common sample-size footing.

    Two consistent estimators of the same parameter, each with
    sqrt(n)-limit variance v, are compared by the ratio of the sample
    sizes they need for the same limiting precision.  If T1 based on n1
    observations has asymptotic variance var1 and T2 based on n2 has
    var2, the asymptotic relative efficiency of T2 with respect to T1 is

        ARE(T2, T1) = (var1 / n1) / (var2 / n2),

    so a value above one means T2 is the more efficient of the pair.  The
    two classical Gaussian benchmarks follow from this definition: the
    sample median against the mean gives 2/pi = 0.6366, and the
    Hodges-Lehmann estimator (equivalently the Wilcoxon signed-rank test)
    against the mean gives 3/pi = 0.9549.

    Parameters
    ----------
    var1, var2 : float
        Asymptotic variances, strictly positive.
    n1, n2 : float
        Sample sizes the variances refer to.

    Returns
    -------
    RichResult
        ``are``, ``logare``, ``var1``, ``var2``, ``normalmedian``,
        ``normalhl``.

    References
    ----------
    Hodges, J. L. and Lehmann, E. L. (1956), "The efficiency of some
    nonparametric competitors of the t-test", Annals of Mathematical
    Statistics 27(2), 324-335, which is the source of the 3/pi figure for
    the Wilcoxon/Hodges-Lehmann procedure at the normal and of the
    general variance-ratio definition of efficiency used here.  Standard
    published form; the Annals article was not in the local corpus and
    was not read for this implementation.  The reported benchmarks 2/pi
    and 3/pi are closed forms, not values recalled from the paper.
    """
    v1, v2 = float(var1), float(var2)
    m1, m2 = float(n1), float(n2)
    if v1 <= 0.0 or v2 <= 0.0:
        raise ValueError("variances must be strictly positive")
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("sample sizes must be strictly positive")
    are = (v1 / m1) / (v2 / m2)
    return RichResult(payload={
        "are": are, "logare": math.log(are), "var1": v1, "var2": v2,
        "normalmedian": 2.0 / math.pi, "normalhl": 3.0 / math.pi,
        "method": "Asymptotic relative efficiency (Hodges-Lehmann 1956)"})


asymptotic_relative_efficiency = areratio


def cheatsheet():
    return "areN: Asymptotic relative efficiency of two estimators."
