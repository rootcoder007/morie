"""Family-wise error rate of M tests (ESL sec 18.7)."""

from ._richresult import RichResult

__all__ = ["esl_fwer"]


def esl_fwer(alpha, M):
    r"""FWER of M independent level-:math:`\alpha` tests and the per-test levels that control it.

    ESL sec 18.7.1: with independent tests the family-wise error rate is
    :math:`1 - (1-\alpha)^M`; positive dependence makes it smaller, and the
    Bonferroni bound :math:`\min(1, M\alpha)` holds under any dependence. To hold
    the FWER at :math:`\alpha` each test is run at :math:`\alpha/M` (Bonferroni)
    or :math:`1 - (1-\alpha)^{1/M}` (Sidak, exact for independent tests).

    Parameters
    ----------
    alpha : float
        Level, in (0, 1).
    M : int
        Number of tests, >= 1.

    Returns
    -------
    RichResult
        ``fwer_independent``, ``bonferroni_bound``, ``per_test_bonferroni``,
        ``per_test_sidak``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 18.7.
    """
    alpha, M = float(alpha), int(M)
    if not 0 < alpha < 1 or M < 1:
        raise ValueError("need 0 < alpha < 1 and M >= 1")
    f = 1 - (1 - alpha) ** M
    return RichResult(
        title="Family-wise error rate",
        summary_lines=[("fwer_independent", f)],
        payload={
            "fwer_independent": f,
            "bonferroni_bound": min(1.0, M * alpha),
            "per_test_bonferroni": alpha / M,
            "per_test_sidak": 1 - (1 - alpha) ** (1 / M),
        },
    )


def cheatsheet():
    return "eslfwe: FWER = 1 - (1 - alpha)^M; per-test alpha/M or 1 - (1 - alpha)^(1/M)"
