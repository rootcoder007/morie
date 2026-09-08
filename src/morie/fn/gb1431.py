# morie.fn -- function file (rootcoder007/morie)
"""Special results for k x 2 contingency tables -- eq. (14.3.2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['chik2', 'gibbons_k2_contingency']


def chik2(successes, ns):
    """Test of equal proportions across k groups, Sec. 14.3.

    Book p. 513.  With Y_i successes out of n_i in group i and
    p-hat the pooled proportion, eq. (14.3.2) gives the simplified
    computing form of the goodness-of-fit criterion (14.3.1),

    .. math:: Q = \\frac{1}{\\hat p(1-\\hat p)}
        \\sum_{i=1}^{k}\\frac{Y_i^2}{n_i}
        - \\frac{N\\hat p}{1-\\hat p},

    asymptotically chi-square with k - 1 degrees of freedom.  The
    book's Example 14.3.1 (responses 125, 81, 40 out of 200, 200, 200)
    gives Q = 74.70 on 2 degrees of freedom.

    Parameters
    ----------
    successes : sequence of float
        Y_1, ..., Y_k.
    ns : sequence of int
        The k group sizes.

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``phat``,
        ``props``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.3, eq. (14.3.2), p. 514.
    """
    y = [float(v) for v in successes]
    nv = [float(v) for v in ns]
    k = len(y)
    if k < 2 or len(nv) != k:
        raise ValueError("need at least 2 groups and matching sizes.")
    if any(v <= 0 for v in nv):
        raise ValueError("group sizes must be positive.")
    nn = sum(nv)
    ph = sum(y) / nn
    if not 0.0 < ph < 1.0:
        raise ValueError("the pooled proportion must lie inside (0, 1).")
    q = sum(y[i] ** 2 / nv[i] for i in range(k)) / (ph * (1.0 - ph)) - (
        nn * ph / (1.0 - ph)
    )
    df = k - 1
    return RichResult(
        payload={
            "statistic": float(q),
            "df": int(df),
            "p_value": float(stats.chi2.sf(q, df)),
            "phat": float(ph),
            "props": [y[i] / nv[i] for i in range(k)],
            "k": int(k),
            "n": float(nn),
            "method": "k x 2 equal-proportions test, eq. (14.3.2)",
        }
    )


gibbons_k2_contingency = chik2
