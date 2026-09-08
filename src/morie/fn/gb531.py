# morie.fn -- function file (rootcoder007/morie)
"""Distribution-free test for a population quantile."""

import math

from ._richresult import RichResult

__all__ = ['quanttest', 'gibbons_quantile_test']


def quanttest(x, q0, p=0.5, alternative="two-sided"):
    """Binomial test of H0: x_p = q0 using the count below q0.

    Section 5.3 (book p. 163): under H0 the number of sample values
    not exceeding q0,

    .. math:: K = \\#\\{X_i \\le q_0\\},

    is Binomial(n, p), so an exact test needs no assumption beyond
    continuity.  ``alternative`` is one of ``"two-sided"``,
    ``"less"`` (H1: x_p < q0, large K) or ``"greater"``
    (H1: x_p > q0, small K).

    Parameters
    ----------
    x : sequence of float
        The sample, n >= 1.
    q0 : float
        Hypothesised quantile value.
    p : float, optional
        Quantile level (default 0.5, the median).
    alternative : str, optional
        Direction of the alternative.

    Returns
    -------
    RichResult
        keys ``statistic`` (K), ``p_value``, ``n``, ``p``, ``mean``,
        ``var``, ``alternative``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.3, p. 163.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    p = float(p)
    if n < 1:
        raise ValueError("x must be non-empty.")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1).")
    k = sum(1 for v in xs if v <= float(q0))

    def _pmf(i):
        return math.comb(n, i) * p**i * (1.0 - p) ** (n - i)

    lower = sum(_pmf(i) for i in range(k + 1))
    upper = sum(_pmf(i) for i in range(k, n + 1))
    if alternative == "less":
        pv = lower
    elif alternative == "greater":
        pv = upper
    elif alternative == "two-sided":
        pv = min(1.0, 2.0 * min(lower, upper))
    else:
        raise ValueError("alternative must be two-sided, less or greater.")
    return RichResult(
        payload={
            "statistic": int(k),
            "p_value": float(pv),
            "n": n,
            "p": p,
            "mean": n * p,
            "var": n * p * (1.0 - p),
            "alternative": alternative,
            "method": "quantile test: K = #{X_i <= q0} ~ Bin(n, p)",
        }
    )


gibbons_quantile_test = quanttest
