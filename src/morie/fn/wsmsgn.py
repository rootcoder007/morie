# morie.fn -- function file (rootcoder007/morie)
"""Sign test for a median."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgntest", "wasserman_sign_test"]


def sgntest(x, md=0.0):
    """Exact sign test of H0: median = md.

    The sign test throws away the magnitudes and keeps only the signs,
    which costs power but buys an EXACT p-value under nothing more than
    continuity -- no symmetry, no normality, no finite variance.  Ties
    at md carry no information about direction and are discarded, which
    reduces the effective n; that reduced n is returned rather than the
    input length.

    Formula: S = #{x_i > md}, m = #{x_i != md};
             S ~ Binomial(m, 1/2) under H0;
             two-sided p = 2 P(Bin(m, 1/2) >= max(S, m - S)), capped at 1

    Parameters
    ----------
    x : array-like
        The sample.
    md : float
        Null median.

    Returns
    -------
    RichResult
        ``statistic`` (S), ``p_value``, ``n_effective``, ``n_ties``,
        ``estimate`` (sample median), ``n``.

    References
    ----------
    Dixon & Mood (1946), The statistical sign test, Journal of the
    American Statistical Association 41(236), 557-566 -- the primary
    source.  Wasserman (2004), All of Statistics, does NOT contain the
    sign test; the full text of the book was fetched and searched to
    establish that, so it is not cited for this formula.
    """
    x = C.vec(x)
    n = len(x)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    md = float(md)
    pos = sum(1 for v in x if v > md)
    neg = sum(1 for v in x if v < md)
    m = pos + neg
    if m == 0:
        raise ValueError("every observation equals md; the test is vacuous")
    k = max(pos, neg)
    # Exact upper tail of Binomial(m, 1/2), computed in logs so that a
    # large m does not overflow the binomial coefficient.
    tail = 0.0
    for j in range(k, m + 1):
        lg = (math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1)
              - m * math.log(2.0))
        tail += math.exp(lg)
    s = sorted(x)
    med = (s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2]))
    return RichResult(payload={
        "statistic": float(pos), "p_value": min(1.0, 2.0 * tail),
        "n_effective": float(m), "n_ties": float(n - m), "estimate": med,
        "n": float(n), "method": "Exact sign test, Binomial(m, 1/2)"})


wasserman_sign_test = sgntest


def cheatsheet():
    return "wsmsgn: S ~ Bin(m, 1/2); ties at md dropped from m"
