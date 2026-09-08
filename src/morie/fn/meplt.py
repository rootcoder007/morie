# morie.fn -- function file (rootcoder007/morie)
"""Empirical mean excess (mean residual life) function.

Source: Davison, A. C. and Smith, R. L. (1990), "Models for exceedances
over high thresholds", *Journal of the Royal Statistical Society Series
B* 52(3):393-425 (with discussion, to 442).  The mean excess function

    e(u) = E[X - u | X > u]

is estimated by the sample mean of the exceedances,

    e_hat(u) = ( sum_{i: x_i > u} (x_i - u) ) / N_u,   N_u = #{x_i > u}

and plotted against u.  The plot is a threshold-selection diagnostic:
for a generalised Pareto tail with shape xi < 1 the mean excess function
is exactly LINEAR in u above the threshold at which the approximation
holds, with slope xi / (1 - xi), so the analyst reads off the lowest u
beyond which the curve is straight.  That linearity is the whole point
of the display and is stated here so the output is not mistaken for an
estimator of anything on its own.

Pointwise normal confidence limits are returned alongside, using the
standard error of a sample mean over the N_u exceedances,
s_u / sqrt(N_u) with s_u the unbiased sample standard deviation of the
exceedances.  These are the usual bands drawn on a mean residual life
plot; they are pointwise, not simultaneous, and are only trustworthy
where N_u is not small, which is precisely where the right-hand end of
the plot is worst.  ``n_exceed`` is returned so a caller can see that.

The comparison is strict, x_i > u, so a threshold placed exactly on an
observed value excludes it.
"""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["mean_excess"]


def mean_excess(x, u_grid=None, conf_level=0.95):
    """Empirical mean excess over a grid of thresholds.

    Parameters
    ----------
    x : array-like
        Sample.
    u_grid : array-like, optional
        Thresholds.  Defaults to the sorted distinct values of x with
        the largest dropped, which always leaves at least one
        exceedance per threshold.
    conf_level : float
        Level for the pointwise normal limits.

    Returns
    -------
    RichResult
        ``u`` (the grid), ``e`` (mean excess at each u), ``se``,
        ``ci_lower``, ``ci_upper``, ``n_exceed``, ``sd_excess``, ``n``.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("mean excess needs at least two observations")
    if not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must lie strictly between 0 and 1")
    if u_grid is None:
        uniq = sorted(set(xs))
        if len(uniq) < 2:
            raise ValueError("x is constant; no threshold leaves an exceedance")
        grid = uniq[:-1]
    else:
        grid = [float(v) for v in u_grid]
        if len(grid) == 0:
            raise ValueError("u_grid must be non-empty")
    zq = float(stats.norm.ppf(0.5 + 0.5 * float(conf_level)))
    nan = float("nan")
    us = []
    es = []
    ses = []
    los = []
    his = []
    cnt = []
    sds = []
    for u in grid:
        ex = [v - u for v in xs if v > u]
        m = len(ex)
        us.append(u)
        cnt.append(m)
        if m == 0:
            es.append(nan)
            ses.append(nan)
            sds.append(nan)
            los.append(nan)
            his.append(nan)
            continue
        mu = sum(ex) / m
        es.append(mu)
        if m < 2:
            ses.append(nan)
            sds.append(nan)
            los.append(nan)
            his.append(nan)
            continue
        ss = 0.0
        for v in ex:
            dv = v - mu
            ss += dv * dv
        sd = math.sqrt(ss / (m - 1))
        se = sd / math.sqrt(m)
        sds.append(sd)
        ses.append(se)
        los.append(mu - zq * se)
        his.append(mu + zq * se)
    return RichResult(payload={
        "u": [float(v) for v in us], "e": [float(v) for v in es],
        "se": [float(v) for v in ses], "sd_excess": [float(v) for v in sds],
        "ci_lower": [float(v) for v in los],
        "ci_upper": [float(v) for v in his],
        "n_exceed": cnt, "conf_level": float(conf_level), "n": n,
        "method": "Davison & Smith (1990) mean excess e(u)=E[X-u|X>u]; "
                  "linear in u above a generalised Pareto threshold"})


def cheatsheet():
    return "meplt: Davison & Smith (1990) mean excess / mean residual life"


# compact alias per ledger/NAMING.md
meanexcess = mean_excess
