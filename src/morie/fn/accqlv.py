"""Quality levels (AQL, RQL) and average outgoing quality of a single sampling plan."""

from ._richresult import RichResult
from ._stats_core import beta, binom

__all__ = ["acceptance_quality_levels"]


def _golden_max(f, a, b, tol=1e-12):
    g = (5**0.5 - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    while b - a > tol:
        if f(c) > f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return (a + b) / 2


def acceptance_quality_levels(n, c, alpha=0.05, beta_risk=0.10, lot_size=None):
    r"""AQL, RQL and AOQL of the binomial single sampling plan (n, c).

    A lot is accepted when a sample of n holds at most c defectives, with
    probability :math:`P_A(p) = \sum_{d=0}^{c}\binom{n}{d}p^d(1-p)^{n-d}`.
    Hedderich, Sachs & Reynarowych (2023, eqs 7.17-7.18): AQL solves
    :math:`P_A = 1-\alpha`, RQL solves :math:`P_A = \beta`; since
    :math:`P_A(p) = 1 - I_p(c+1, n-c)` both are beta quantiles. The average
    outgoing quality :math:`AOQ(p) = p P_A(p)(N-n)/N` is maximised for AOQL
    (golden section; :math:`(N-n)/N = 1` without ``lot_size``).

    Parameters
    ----------
    n, c : int
        Sample size and acceptance number (0 <= c < n).
    alpha, beta_risk : float
        Producer's and consumer's risks.
    lot_size : int, optional
        Lot size N.

    Returns
    -------
    RichResult
        ``aql``, ``rql``, ``aoql``, ``p_aoql``.

    References
    ----------
    Montgomery, D. C. (2013). Introduction to Statistical Quality Control
    (7th ed.), ch. 15.
    """
    n, c = int(n), int(c)
    if not 0 <= c < n:
        raise ValueError("need 0 <= c < n")
    f = 1.0 if lot_size is None else (lot_size - n) / lot_size
    aql = float(beta.ppf(alpha, c + 1, n - c))
    rql = float(beta.ppf(1 - beta_risk, c + 1, n - c))
    aoq = lambda p: p * float(binom.cdf(c, n, p)) * f  # noqa: E731
    pm = _golden_max(aoq, 0.0, 1.0)
    return RichResult(
        title="Sampling plan quality levels",
        summary_lines=[("aql", aql), ("rql", rql), ("aoql", aoq(pm))],
        payload={"aql": aql, "rql": rql, "aoql": aoq(pm), "p_aoql": pm},
    )


def cheatsheet():
    return "accqlv: AQL = qbeta(alpha, c + 1, n - c), RQL = qbeta(1 - beta, c + 1, n - c), AOQL = max p P_A (N - n)/N"
