# morie.fn -- function file (rootcoder007/morie)
"""Anderson-Darling goodness-of-fit statistic W_n^2."""

import math

from ._richresult import RichResult

__all__ = ['adtest', 'gibbons_anderson_darling']


_AD_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.15)
_AD_TABLE = {
    "specified": (3.857, 3.070, 2.492, 1.933, 1.610),
    "normal-mean": (1.551, 1.285, 1.087, 0.894, 0.782),
    "normal-var": (3.702, 2.898, 2.308, 1.743, 1.430),
    "normal-both": (1.035, 0.873, 0.752, 0.631, 0.561),
    "exponential": (1.959, 1.591, 1.321, 1.062, 0.916),
}


def adtest(x, cdf, case="specified", alpha=0.05):
    """Anderson-Darling statistic W_n^2 and Stephens's modified form.

    Section 4.7 (book p. 138), eq. (4.7.1).  Weighting the squared EDF
    deviation by 1/[F_0(1 - F_0)] emphasises the tails, and the
    resulting statistic collapses to

    .. math:: W_n^2 = -n - \\frac{1}{n}\\sum_{j=1}^{n}(2j-1)
        \\bigl[\\ln Z_j + \\ln(1 - Z_{n-j+1})\\bigr],
        \\qquad Z_j = F_0(X_{(j)}).

    Table 4.7.1 (book p. 140, from Stephens 1986) gives the upper tail
    percentage points for each null situation, together with the finite
    sample modification A* each one needs.  ``"specified"``,
    ``"normal-mean"`` and ``"normal-var"`` take A* = W^2 unmodified;
    ``"normal-both"`` uses W^2 (1 + 0.75/n + 2.25/n^2) and
    ``"exponential"`` uses W^2 (1 + 0.3/n).

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    cdf : callable
        The hypothesised continuous cdf F_0 (with any unknown
        parameters already replaced by their maximum likelihood
        estimates, as Sec. 4.7 prescribes).
    case : str, optional
        Row of Table 4.7.1 to use (default ``"specified"``).
    alpha : float, optional
        One of 0.01, 0.025, 0.05, 0.10, 0.15 (default 0.05).

    Returns
    -------
    RichResult
        keys ``statistic`` (W_n^2), ``astar``, ``crit``, ``reject``,
        ``z``, ``n``, ``case``, ``alpha``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.7, eq. (4.7.1), p. 138, and
    Table 4.7.1, p. 140 (adapted from Stephens, M. A., 1986, Tests
    Based on EDF Statistics, in Goodness-of-Fit Techniques, Marcel
    Dekker).
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if case not in _AD_TABLE:
        raise ValueError("case must be a row label of Table 4.7.1.")
    alpha = float(alpha)
    if alpha not in _AD_LEVELS:
        raise ValueError("alpha must be one of 0.01, 0.025, 0.05, 0.10, 0.15.")
    z = [float(cdf(v)) for v in xs]
    if any(v <= 0.0 or v >= 1.0 for v in z):
        raise ValueError("F_0 values must lie strictly inside (0, 1).")
    s = 0.0
    for j in range(1, n + 1):
        s += (2.0 * j - 1.0) * (
            math.log(z[j - 1]) + math.log(1.0 - z[n - j])
        )
    a2 = -n - s / n
    if case == "normal-both":
        astar = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    elif case == "exponential":
        astar = a2 * (1.0 + 0.3 / n)
    else:
        astar = a2
    crit = _AD_TABLE[case][_AD_LEVELS.index(alpha)]
    return RichResult(
        payload={
            "statistic": float(a2),
            "astar": float(astar),
            "crit": float(crit),
            "reject": int(astar > crit),
            "z": z,
            "n": n,
            "case": case,
            "alpha": alpha,
            "method": "Anderson-Darling W_n^2, eq. (4.7.1), Table 4.7.1",
        }
    )


gibbons_anderson_darling = adtest
