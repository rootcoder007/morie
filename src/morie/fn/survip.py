# morie.fn -- function file (rootcoder007/morie)
"""Design-corrected p-value for a test statistic computed as if simple random."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["survey_p_value"]


def survey_p_value(test_stat, DEFF=1.0, df=1):
    """Divide a naive chi-square statistic by its design effect.

    A test statistic computed from complex-survey data while pretending
    the sample was simple and random is inflated by roughly the design
    effect, and its nominal p-value is correspondingly too small.  The
    first-order correction is a single division; it is what Korn and
    Graubard recommend when only a summary design effect is available,
    and it is exactly Rao and Scott's first-order adjustment when the
    design effect used is the mean generalized design effect.

    Formula: ``X2_adj = X2 / deff``, ``p = P(chi2_df > X2_adj)``.

    Parameters
    ----------
    test_stat : float
        Naive chi-square statistic, non-negative.
    DEFF : float, default 1
        Design effect, positive.  ``1`` returns the uncorrected p-value.
    df : int, default 1
        Degrees of freedom.

    Returns
    -------
    RichResult
        ``estimate`` (corrected p-value), ``p_naive``, ``statistic``
        (corrected statistic), ``statistic_naive``, ``deff``, ``df``,
        ``inflation`` (ratio of the two p-values), ``method``.

    References
    ----------
    Korn, E. L. & Graubard, B. I. (1999).  Analysis of Health Surveys.
    Wiley, chapter 3 (design effects and the adjustment of test
    statistics).  <https://doi.org/10.1002/9781118032619>
    Rao, J. N. K. & Scott, A. J. (1984).  The Annals of Statistics
    12(1):46-60.  <https://doi.org/10.1214/aos/1176346391>
    """
    x = float(test_stat)
    d = float(DEFF)
    k = int(df)
    if x < 0.0:
        raise ValueError("survey_p_value: test_stat must be non-negative")
    if d <= 0.0:
        raise ValueError("survey_p_value: DEFF must be positive")
    if k < 1:
        raise ValueError("survey_p_value: df must be at least 1")
    adj = x / d
    p0 = 1.0 - C.pchisq(x, k)
    p1 = 1.0 - C.pchisq(adj, k)
    return RichResult(payload={
        "estimate": float(p1), "p_naive": float(p0), "statistic": float(adj),
        "statistic_naive": float(x), "deff": d, "df": k,
        "inflation": float(p1 / p0) if p0 > 0.0 else float("inf"),
        "method": "first-order design correction X2/deff [Korn & Graubard 1999]"})


# CANONICAL TEST
# >>> r = survey_p_value(3.841458820694124, 1.0, 1)
# >>> assert abs(r["estimate"] - 0.05) < 1e-9      # qchisq(.95, 1) inverts to .05
# >>> assert abs(r["estimate"] - r["p_naive"]) < 1e-15
# >>> assert survey_p_value(3.841458820694124, 2.0, 1)["estimate"] > 0.05


def cheatsheet():
    return "survip(test_stat, DEFF, df): design-corrected chi-square p-value."
