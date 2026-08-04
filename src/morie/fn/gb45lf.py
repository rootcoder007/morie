# morie.fn -- function file (rootcoder007/morie)
"""Lilliefors test for normality with estimated mean and variance."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['lillienorm', 'gibbons_lilliefors_normal']


_TABLE_O = {
    4: (0.344, 0.375, 0.414, 0.432), 5: (0.320, 0.344, 0.398, 0.427),
    6: (0.298, 0.323, 0.369, 0.421), 7: (0.281, 0.305, 0.351, 0.399),
    8: (0.266, 0.289, 0.334, 0.383), 9: (0.252, 0.273, 0.316, 0.366),
    10: (0.240, 0.261, 0.305, 0.350), 11: (0.231, 0.251, 0.291, 0.331),
    12: (0.223, 0.242, 0.281, 0.327), 14: (0.208, 0.226, 0.262, 0.302),
    16: (0.195, 0.213, 0.249, 0.291), 18: (0.185, 0.201, 0.234, 0.272),
    20: (0.176, 0.192, 0.223, 0.266), 25: (0.159, 0.173, 0.202, 0.236),
    30: (0.146, 0.159, 0.186, 0.219), 40: (0.127, 0.139, 0.161, 0.190),
    50: (0.114, 0.125, 0.145, 0.173), 60: (0.105, 0.114, 0.133, 0.159),
    75: (0.094, 0.102, 0.119, 0.138), 100: (0.082, 0.089, 0.104, 0.121),
}
_TABLE_O_TAIL = (0.816, 0.888, 1.038, 1.212)
_LEVELS = (0.100, 0.050, 0.010, 0.001)


def lillienorm(x, alpha=0.05):
    """Lilliefors's KS test for normality, parameters estimated.

    Section 4.5 (book p. 126).  The statistic is the ordinary KS
    statistic computed against the normal cdf with the sample mean and
    the sample standard deviation (divisor n - 1) substituted for the
    unknown parameters,

    .. math:: D = \\sup_x |S_n(x) - \\Phi((x - \\bar X)/s)|,

    but the null distribution is NOT that of D_n, because the
    parameters were estimated; critical values come from Table O
    (book p. 589), which is reproduced here verbatim.  Sample sizes
    above 100 use the table's own large-sample row, c/sqrt(N).

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 4.
    alpha : float, optional
        One of 0.10, 0.05, 0.01, 0.001 (default 0.05).

    Returns
    -------
    RichResult
        keys ``statistic``, ``dcrit``, ``reject`` (1/0), ``mean``,
        ``sd``, ``n``, ``n_table`` (the table row used), ``alpha``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.5, p. 126; Table O, p. 589
    (adapted from Edgeman and Scott, 1987).
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 4:
        raise ValueError("need at least 4 observations.")
    alpha = float(alpha)
    if alpha not in _LEVELS:
        raise ValueError("alpha must be one of 0.10, 0.05, 0.01, 0.001.")
    col = _LEVELS.index(alpha)
    mu = sum(xs) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in xs) / (n - 1.0))
    if sd <= 0.0:
        raise ValueError("sample has zero variance.")
    z = [stats.norm.cdf((v - mu) / sd) for v in xs]
    d = max(
        max((i + 1) / n - z[i], z[i] - i / n) for i in range(n)
    )
    if n > 100:
        ntab = 0
        dcrit = _TABLE_O_TAIL[col] / math.sqrt(n)
    else:
        keys = sorted(k for k in _TABLE_O if k <= n)
        ntab = keys[-1] if keys else 4
        dcrit = _TABLE_O[ntab][col]
    return RichResult(
        payload={
            "statistic": float(d),
            "dcrit": float(dcrit),
            "reject": int(d > dcrit),
            "mean": float(mu),
            "sd": float(sd),
            "n": n,
            "n_table": int(ntab),
            "alpha": alpha,
            "method": "Lilliefors test for normality, Table O",
        }
    )


gibbons_lilliefors_normal = lillienorm
