# morie.fn -- function file (rootcoder007/morie)
"""Fisher's exact test for a 2 x 2 table."""

import math

from ._richresult import RichResult

__all__ = ['fisherex', 'gibbons_fisher_exact']


def _hyper(a, r1, r2, c1):
    """P(A = a) for the hypergeometric null of a 2 x 2 table."""
    nn = r1 + r2
    if a < max(0, c1 - r2) or a > min(r1, c1):
        return 0.0
    return (
        math.comb(r1, a) * math.comb(r2, c1 - a) / math.comb(nn, c1)
    )


def fisherex(table, alternative="two-sided"):
    """Exact conditional test of independence, Sec. 14.4.

    Book p. 517.  Conditioning on both margins, the (1,1) count A of a
    2 x 2 table has the hypergeometric null distribution

    .. math:: P(A = a) = \\frac{\\binom{r_1}{a}
        \\binom{r_2}{c_1-a}}{\\binom{N}{c_1}},

    so a p-value can be computed exactly with no large-sample
    approximation.  The two-sided p-value sums every table whose
    probability does not exceed that of the observed one -- the
    convention that keeps the test conditional rather than doubling a
    tail.

    Parameters
    ----------
    table : sequence of sequence of float
        The 2 x 2 table [[a, b], [c, d]].
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` or ``"less"``.

    Returns
    -------
    RichResult
        keys ``p_value``, ``p_greater``, ``p_less``, ``prob``
        (probability of the observed table), ``statistic`` (a),
        ``support``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.4, p. 517 (Fisher, 1934).
    """
    tb = [[int(round(float(v))) for v in row] for row in table]
    if len(tb) != 2 or any(len(row) != 2 for row in tb):
        raise ValueError("table must be 2 x 2.")
    a, b = tb[0]
    c, d = tb[1]
    if min(a, b, c, d) < 0:
        raise ValueError("counts must be non-negative.")
    r1 = a + b
    r2 = c + d
    c1 = a + c
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    if hi < lo:
        raise ValueError("degenerate margins.")
    probs = {k: _hyper(k, r1, r2, c1) for k in range(lo, hi + 1)}
    pobs = probs[a]
    pg = sum(p for k, p in probs.items() if k >= a)
    pl = sum(p for k, p in probs.items() if k <= a)
    if alternative == "greater":
        pv = pg
    elif alternative == "less":
        pv = pl
    elif alternative == "two-sided":
        pv = sum(p for p in probs.values() if p <= pobs * (1.0 + 1e-12))
    else:
        raise ValueError("alternative must be two-sided, greater or less.")
    return RichResult(
        payload={
            "p_value": float(min(1.0, pv)),
            "p_greater": float(pg),
            "p_less": float(pl),
            "prob": float(pobs),
            "statistic": int(a),
            "support": [lo, hi],
            "method": "Fisher exact test, hypergeometric null (Sec. 14.4)",
        }
    )


gibbons_fisher_exact = fisherex
