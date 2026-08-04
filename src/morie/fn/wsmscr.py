# morie.fn -- function file (rootcoder007/morie)
"""Rao score test for a binomial proportion."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["scoretest", "wasserman_score_test"]


def scoretest(successes, n, p0=0.5):
    """Score (Rao) test of H0: p = p0 for a binomial proportion.

    The score test differs from the Wald test in exactly one place, and
    it is the place that matters: the variance is evaluated at the NULL
    p0, not at the estimate.  That is why the score test still works
    when phat is 0 or 1 -- where the Wald standard error collapses to
    zero and the Wald test cannot be computed at all.

    Formula: U = (S - n p0) / sqrt(n p0 (1 - p0));
             U^2 ~ chi^2_1;  p = 2(1 - Phi(|U|))

    Parameters
    ----------
    successes : int
        Number of successes S.
    n : int
        Number of trials.
    p0 : float
        Null proportion, strictly between 0 and 1.

    Returns
    -------
    RichResult
        ``statistic`` (U), ``chisq`` (U^2), ``p_value``, ``estimate``
        (phat), ``se_null``, ``n``.

    References
    ----------
    Rao (1948), Large sample tests of statistical hypotheses concerning
    several parameters with applications to problems of estimation,
    Mathematical Proceedings of the Cambridge Philosophical Society
    44(1), 50-57 -- the primary source for the score test.  Wasserman
    (2004), All of Statistics, treats the Wald test (Definition 10.3),
    the chi-square test and the likelihood ratio test (Definition
    10.21) but does NOT give the score test, so it is not cited for
    this formula; the full text of the book was fetched and searched to
    establish that.
    """
    n = int(n)
    S = float(successes)
    p0 = float(p0)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < S <= n and S != 0.0:
        raise ValueError("successes must lie in 0..n")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must lie strictly between 0 and 1")
    se = math.sqrt(n * p0 * (1.0 - p0))
    U = (S - n * p0) / se
    return RichResult(payload={
        "statistic": U, "chisq": U * U,
        "p_value": 2.0 * (1.0 - C.pnorm(abs(U))), "estimate": S / n,
        "se_null": se, "n": float(n),
        "method": "Rao score test for a binomial proportion"})


wasserman_score_test = scoretest


def cheatsheet():
    return "wsmscr: U = (S - n p0)/sqrt(n p0 (1-p0)); variance at the NULL"
