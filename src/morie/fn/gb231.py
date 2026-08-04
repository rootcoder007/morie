# morie.fn -- function file (rootcoder007/morie)
"""EDF count nS_n(x) is Binomial(n, F(x)) -- Gibbons Theorem 2.3.1."""

import math

from ._richresult import RichResult

__all__ = ['edfbinom', 'gibbons_edf_binomial']


def edfbinom(n, fx, i=None):
    """Distribution of the EDF count T_n(x) = n * S_n(x).

    Theorem 2.3.1 (book p. 33): for fixed x, the number of sample
    values not exceeding x is Binomial(n, F_X(x)),

    .. math:: P[n S_n(x) = i] = \\binom{n}{i} F(x)^i [1 - F(x)]^{n-i}.

    Parameters
    ----------
    n : int
        Sample size, n >= 1.
    fx : float
        The population cdf value F_X(x), in [0, 1].
    i : int, optional
        Count at which to evaluate the pmf and cdf.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``pmf``, ``cdf``, ``n``, ``fx``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 2.3.1, eq. (2.3.2), p. 33.
    """
    n = int(n)
    fx = float(fx)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 0.0 <= fx <= 1.0:
        raise ValueError("fx must lie in [0, 1].")
    pmf = float("nan")
    cdf = float("nan")
    if i is not None:
        i = int(i)
        if not 0 <= i <= n:
            raise ValueError("i must lie in 0..n.")
        pmf = math.comb(n, i) * fx**i * (1.0 - fx) ** (n - i)
        cdf = sum(math.comb(n, k) * fx**k * (1.0 - fx) ** (n - k) for k in range(i + 1))
    return RichResult(
        payload={
            "mean": n * fx,
            "var": n * fx * (1.0 - fx),
            "pmf": float(pmf),
            "cdf": float(cdf),
            "n": n,
            "fx": fx,
            "method": "n*S_n(x) ~ Binomial(n, F(x)) (Gibbons Thm 2.3.1)",
        }
    )


gibbons_edf_binomial = edfbinom
