"""Smallest binomial single sampling plan meeting producer and consumer risk points."""

from ._richresult import RichResult
from ._stats_core import binom

__all__ = ["acceptance_sampling_plan"]


def acceptance_sampling_plan(aql, rql, alpha=0.05, beta_risk=0.10, max_n=100000):
    r"""Sample size n and acceptance number c with :math:`P_A(AQL) \ge 1-\alpha`, :math:`P_A(RQL) \le \beta`.

    Solves eq (7.17) of Hedderich, Sachs & Reynarowych (2023) for (n, c) by
    the search of ``AcceptanceSampling::find.plan``: starting at c = 0,
    n = 1, raise n while :math:`P_A(RQL) > \beta`, else raise c while
    :math:`P_A(AQL) < 1-\alpha`.

    Parameters
    ----------
    aql, rql : float
        Producer's and consumer's quality levels, 0 < aql < rql < 1.
    alpha, beta_risk : float
    max_n : int
        Search limit.

    Returns
    -------
    RichResult
        ``n``, ``c``, ``p_accept_aql``, ``p_accept_rql``.

    References
    ----------
    Kiermeier, A. (2008). Visualizing and assessing acceptance sampling
    plans: the R package AcceptanceSampling. JSS 26(6).
    """
    if not 0 < aql < rql < 1:
        raise ValueError("need 0 < aql < rql < 1")
    n, c = 1, 0
    while True:
        if float(binom.cdf(c, n, rql)) > beta_risk:
            n += 1
        elif float(binom.cdf(c, n, aql)) < 1 - alpha:
            c += 1
        else:
            break
        if n > max_n:
            raise ValueError("no plan within max_n")
    pa, pr = float(binom.cdf(c, n, aql)), float(binom.cdf(c, n, rql))
    return RichResult(
        title="Single sampling plan",
        summary_lines=[("n", n), ("c", c)],
        payload={"n": n, "c": c, "p_accept_aql": pa, "p_accept_rql": pr},
    )


def cheatsheet():
    return "accpln: smallest (n, c) with P_A(AQL) >= 1 - alpha and P_A(RQL) <= beta (find.plan search)"
