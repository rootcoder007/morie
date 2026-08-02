# morie.fn -- function file (rootcoder007/morie)
"""Theil-Sen slope estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["theil_sen"]


def theil_sen(x, y, alpha=0.05):
    r"""The Theil-Sen estimator: the median of all pairwise slopes,

    .. math:: \hat\beta = \mathop{\mathrm{med}}_{i<j}
              \frac{y_j - y_i}{x_j - x_i},

    with the intercept :math:`\mathrm{med}_i(y_i - \hat\beta x_i)`
    (Theil 1950; Sen 1968). Pairs with :math:`x_i = x_j` are
    excluded, as Sen specifies -- their slope is undefined, and
    silently treating them as 0 or inf biases the median.

    The estimator's breakdown point is :math:`1 - 1/\sqrt2 \approx
    29.3\%`: the median of :math:`\binom n2` slopes fails only once
    the CONTAMINATED PAIRS are a majority, and a fraction
    :math:`\epsilon` of bad points contaminates
    :math:`1 - (1-\epsilon)^2` of the pairs. Sen's Sec. 5 supplies
    the distribution-free confidence interval: the slope's CI is read
    off the ORDERED pairwise slopes at ranks determined by the
    normal approximation to Kendall's tau statistic, so it needs no
    residual variance estimate at all.

    ``morie.fn.sensSlp`` is this estimator applied to a time-indexed
    series and shares this implementation.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations.
    alpha : float, default 0.05
        Miss probability for Sen's confidence interval.

    Returns
    -------
    RichResult
        keys: ``slope``, ``intercept``, ``ci``, ``n_pairs``,
        ``n_tied_x``, ``breakdown``, ``ci_method``, ``n``, ``method``.

    References
    ----------
    Theil, H. (1950), *Proc. KNAW* 53:386-392, 521-525, 1397-1412.
    Sen, P. K. (1968), "Estimates of the regression coefficient based
    on Kendall's tau", *JASA* 63:1379-1389, Secs. 3 and 5.
    """
    from . import _stats_core as stats

    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if xv.size != yv.size:
        raise ValueError(f"x has {xv.size} entries and y has {yv.size}.")
    n = xv.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    a = float(alpha)
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {a}.")
    i, j = np.triu_indices(n, 1)
    dx = xv[j] - xv[i]
    dy = yv[j] - yv[i]
    ok = dx != 0
    n_tied = int(np.sum(~ok))
    if not ok.any():
        raise ValueError("every pair of x values is tied; no slope is "
                         "defined.")
    slopes = np.sort(dy[ok] / dx[ok])
    N = slopes.size
    slope = float(np.median(slopes))
    intercept = float(np.median(yv - slope * xv))
    # Sen (1968) Sec. 5: ranks from the normal approximation to the
    # variance of Kendall's score, ties in x accounted for
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = stats.norm.ppf(1 - a / 2)
    C = z * np.sqrt(var_s)
    m1 = int(np.floor((N - C) / 2))
    m2 = int(np.ceil((N + C) / 2))
    ci = (float(slopes[max(m1, 0)]),
          float(slopes[min(m2, N - 1)]))
    return RichResult(payload={
        "slope": slope, "intercept": intercept, "ci": ci,
        "n_pairs": int(N), "n_tied_x": n_tied,
        "breakdown": 0.29289321881345254,      # 1 - 1/sqrt(2)
        "ci_method": "Sen (1968) Sec. 5: order statistics of the pairwise "
                     "slopes at Kendall-tau ranks; no residual variance "
                     "is estimated",
        "n": int(n),
        "method": "Theil-Sen: median of pairwise slopes, median-residual intercept"})


def cheatsheet():
    return "theils: median of C(n,2) slopes -- breakdown 1 - 1/sqrt(2), CI from order statistics"
