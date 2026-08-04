# morie.fn -- function file (rootcoder007/morie)
"""Lilliefors test for exponentiality with estimated mean."""

import math

from ._richresult import RichResult

__all__ = ['lillieexp', 'gibbons_lilliefors_exp']


_TABLE_T = {
    4: (0.444, 0.483, 0.556, 0.626), 5: (0.405, 0.443, 0.514, 0.585),
    6: (0.374, 0.410, 0.477, 0.551), 7: (0.347, 0.381, 0.444, 0.509),
    8: (0.327, 0.359, 0.421, 0.502), 9: (0.310, 0.339, 0.399, 0.460),
    10: (0.296, 0.325, 0.379, 0.444), 11: (0.284, 0.312, 0.366, 0.433),
    12: (0.271, 0.299, 0.350, 0.412), 14: (0.252, 0.277, 0.325, 0.388),
    16: (0.237, 0.261, 0.311, 0.366), 18: (0.224, 0.247, 0.293, 0.328),
    20: (0.213, 0.234, 0.279, 0.329), 25: (0.192, 0.211, 0.251, 0.296),
    30: (0.176, 0.193, 0.229, 0.270), 40: (0.153, 0.168, 0.201, 0.241),
    50: (0.137, 0.150, 0.179, 0.214), 60: (0.125, 0.138, 0.164, 0.193),
    75: (0.113, 0.124, 0.146, 0.173), 100: (0.098, 0.108, 0.127, 0.150),
}
_TABLE_T_TAIL = (0.980, 1.077, 1.274, 1.501)
_LEVELS_T = (0.100, 0.050, 0.010, 0.001)


def lillieexp(x, alpha=0.05):
    """Lilliefors's KS test for the exponential, mean estimated.

    Section 4.6 (book p. 133).  The statistic is the KS statistic
    against F_0(x) = 1 - exp(-x / xbar), the exponential cdf with the
    sample mean substituted for the unknown mean, and the critical
    values are those of Table T (book p. 598), reproduced here.
    Sample sizes above 100 use the table's own c/sqrt(N) row.

    Parameters
    ----------
    x : sequence of float
        Non-negative sample, n >= 4.
    alpha : float, optional
        One of 0.10, 0.05, 0.01, 0.001 (default 0.05).

    Returns
    -------
    RichResult
        keys ``statistic``, ``dcrit``, ``reject``, ``mean``, ``n``,
        ``n_table``, ``alpha``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.6, p. 133; Table T, p. 598.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 4:
        raise ValueError("need at least 4 observations.")
    if any(v < 0.0 for v in xs):
        raise ValueError("exponential data must be non-negative.")
    alpha = float(alpha)
    if alpha not in _LEVELS_T:
        raise ValueError("alpha must be one of 0.10, 0.05, 0.01, 0.001.")
    col = _LEVELS_T.index(alpha)
    mu = sum(xs) / n
    if mu <= 0.0:
        raise ValueError("sample mean must be strictly positive.")
    z = [1.0 - math.exp(-v / mu) for v in xs]
    d = max(max((i + 1) / n - z[i], z[i] - i / n) for i in range(n))
    if n > 100:
        ntab = 0
        dcrit = _TABLE_T_TAIL[col] / math.sqrt(n)
    else:
        keys = sorted(k for k in _TABLE_T if k <= n)
        ntab = keys[-1] if keys else 4
        dcrit = _TABLE_T[ntab][col]
    return RichResult(
        payload={
            "statistic": float(d),
            "dcrit": float(dcrit),
            "reject": int(d > dcrit),
            "mean": float(mu),
            "n": n,
            "n_table": int(ntab),
            "alpha": alpha,
            "method": "Lilliefors test for exponentiality, Table T",
        }
    )


gibbons_lilliefors_exp = lillieexp
