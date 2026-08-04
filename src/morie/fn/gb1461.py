# morie.fn -- function file (rootcoder007/morie)
"""Goodness-of-fit test for multinomial data."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['multgof', 'gibbons_multinomial_gof']


def multgof(observed, probs, ddof=0):
    """Pearson Q for a multinomial null, Sec. 14.6.

    Book p. 528.  With k categories, hypothesised probabilities
    p_1, ..., p_k summing to 1 and N observations,

    .. math:: Q = \\sum_{i=1}^{k}
        \\frac{(f_i - N p_i)^2}{N p_i},

    asymptotically chi-square on k - 1 degrees of freedom, one fewer
    for each parameter estimated from the sample.  The exact
    multinomial probability of the observed vector is returned too,
    since for small N the chi-square reference is the weaker of the
    two.

    Parameters
    ----------
    observed : sequence of float
        Cell counts, k >= 2.
    probs : sequence of float
        Null probabilities, summing to 1.
    ddof : int, optional
        Parameters estimated from the data (default 0).

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``expected``,
        ``prob`` (exact multinomial probability), ``n``, ``k``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.6, p. 528.
    """
    o = [float(v) for v in observed]
    p = [float(v) for v in probs]
    k = len(o)
    if k < 2 or len(p) != k:
        raise ValueError("need at least 2 matching categories.")
    if abs(sum(p) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1.")
    if any(v <= 0.0 for v in p):
        raise ValueError("probs must be strictly positive.")
    nn = sum(o)
    exp = [nn * v for v in p]
    q = sum((o[i] - exp[i]) ** 2 / exp[i] for i in range(k))
    df = k - 1 - int(ddof)
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1.")
    ni = int(round(nn))
    lp = math.lgamma(ni + 1.0)
    for i in range(k):
        ci = int(round(o[i]))
        lp += ci * math.log(p[i]) - math.lgamma(ci + 1.0)
    return RichResult(
        payload={
            "statistic": float(q),
            "df": int(df),
            "p_value": float(stats.chi2.sf(q, df)),
            "expected": exp,
            "prob": float(math.exp(lp)),
            "n": float(nn),
            "k": int(k),
            "method": "multinomial goodness of fit (Sec. 14.6)",
        }
    )


gibbons_multinomial_gof = multgof
