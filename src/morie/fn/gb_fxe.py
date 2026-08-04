# morie.fn -- function file (rootcoder007/morie)
"""One-sided Fisher exact test for a 2 x 2 table."""

import math

from ._richresult import RichResult

__all__ = ['fisherex1', 'gibbons_fisher_one_sided']


def fisherex1(table, alternative="greater"):
    """The single-tail form of the exact conditional test, Sec. 14.4.

    Book p. 517.  When the research hypothesis names a direction, the
    p-value is one hypergeometric tail,

    .. math:: P(A \\ge a) = \\sum_{j \\ge a}
        \\frac{\\binom{r_1}{j}\\binom{r_2}{c_1-j}}
             {\\binom{N}{c_1}},

    or the corresponding lower tail.  Both tails and the point
    probability are returned, so the various two-sided conventions can
    be assembled by the caller rather than being fixed here.

    Parameters
    ----------
    table : sequence of sequence of float
        The 2 x 2 table [[a, b], [c, d]].
    alternative : str, optional
        ``"greater"`` (default) or ``"less"``.

    Returns
    -------
    RichResult
        keys ``p_value``, ``p_greater``, ``p_less``, ``prob``,
        ``statistic``, ``mean``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.4, p. 517.
    """
    from .gb1441 import _hyper

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
    nn = r1 + r2
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    probs = {k: _hyper(k, r1, r2, c1) for k in range(lo, hi + 1)}
    pg = sum(p for k, p in probs.items() if k >= a)
    pl = sum(p for k, p in probs.items() if k <= a)
    if alternative == "greater":
        pv = pg
    elif alternative == "less":
        pv = pl
    else:
        raise ValueError("alternative must be greater or less.")
    return RichResult(
        payload={
            "p_value": float(min(1.0, pv)),
            "p_greater": float(pg),
            "p_less": float(pl),
            "prob": float(probs[a]),
            "statistic": int(a),
            "mean": float(r1 * c1 / nn),
            "method": "one-sided Fisher exact test (Sec. 14.4)",
        }
    )


gibbons_fisher_one_sided = fisherex1
