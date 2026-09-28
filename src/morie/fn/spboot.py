# morie.fn -- function file (rootcoder007/morie)
"""Resampling for spatially dependent data: the moving-block bootstrap of a grid, the stationary
bootstrap of a transect or series, pointwise and simultaneous bootstrap bands, and the toroidal-shift
test of association between two maps."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["block_bootstrap_grid", "stationary_bootstrap", "bootstrap_bands", "toroidal_shift_test"]


def _q7(s, p):
    s = sorted(s)
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _mean(v):
    return ssum(v) / len(v)


def block_bootstrap_grid(
    grid, block: int, *, nboot: int = 200, seed: int = 1, statistic=None, alpha: float = 0.05
) -> RichResult:
    r"""Moving-block bootstrap of a gridded field (Hall 1985; Lahiri 2003).

    The ``nrow x ncol`` grid is rebuilt from ``block x block`` blocks whose
    upper-left corners are drawn uniformly among all overlapping positions
    (Philox stream ``r`` for replicate ``r``), tiled in raster order and
    truncated to the grid size, which preserves dependence within blocks.
    ``statistic`` maps the flattened grid to a number (default the mean).
    Returns the replicates, the bootstrap standard error and the percentile
    interval.

    References
    ----------
    Hall, P. (1985). Resampling a coverage pattern. *Stochastic Processes and
    their Applications*, 20(2), 231-246.
    Lahiri, S. N. (2003). *Resampling Methods for Dependent Data*. Springer.

    Examples
    --------
    >>> r = block_bootstrap_grid([[1.0, 1.0], [1.0, 1.0]], 1, nboot=5)
    >>> r.se
    0.0
    """
    G = [[float(v) for v in row] for row in grid]
    nr, nc = len(G), len(G[0])
    stat = statistic or _mean
    pr, pc = nr - block + 1, nc - block + 1
    br, bc = math.ceil(nr / block), math.ceil(nc / block)
    reps = []
    for r in range(nboot):
        u = [float(v) for v in random_uniform(2 * br * bc, seed=seed, stream=r)]
        out = [[0.0] * nc for _ in range(nr)]
        k = 0
        for bi in range(br):
            for bj in range(bc):
                i0 = min(int(u[2 * k] * pr), pr - 1)
                j0 = min(int(u[2 * k + 1] * pc), pc - 1)
                k += 1
                for a in range(block):
                    for b in range(block):
                        x, y = bi * block + a, bj * block + b
                        if x < nr and y < nc:
                            out[x][y] = G[i0 + a][j0 + b]
        reps.append(stat([v for row in out for v in row]))
    m = _mean(reps)
    se = math.sqrt(ssum((v - m) ** 2 for v in reps) / (nboot - 1))
    return RichResult(
        payload={
            "replicates": reps,
            "se": se,
            "ci": [_q7(reps, alpha / 2), _q7(reps, 1 - alpha / 2)],
            "estimate": stat([v for row in G for v in row]),
        }
    )


def stationary_bootstrap(
    x, p: float, *, nboot: int = 200, seed: int = 1, statistic=None, alpha: float = 0.05
) -> RichResult:
    r"""Stationary bootstrap of a transect or series (Politis and Romano 1994).

    Each replicate concatenates blocks of the circularly wrapped series
    starting at uniform positions with geometric lengths of mean ``1 / p``:
    after each observation a new block starts with probability ``p``
    (Philox stream ``r``; uniforms ``2t`` and ``2t + 1`` for step ``t``).
    The resampled series is stationary. Returns the replicate statistics
    (default the mean), the standard error, the percentile interval and the
    index sequences.

    References
    ----------
    Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap.
    *Journal of the American Statistical Association*, 89(428), 1303-1313.

    Examples
    --------
    >>> r = stationary_bootstrap([1.0, 2.0, 3.0, 4.0], 1e-12, nboot=3)
    >>> all(sorted(ix) == [0, 1, 2, 3] for ix in r.indices)
    True
    """
    xv = [float(v) for v in x]
    n = len(xv)
    stat = statistic or _mean
    reps, idx = [], []
    for r in range(nboot):
        u = [float(v) for v in random_uniform(2 * n, seed=seed, stream=r)]
        ix = []
        cur = min(int(u[0] * n), n - 1)
        for t in range(n):
            if t > 0:
                cur = min(int(u[2 * t + 1] * n), n - 1) if u[2 * t] < p else (cur + 1) % n
            ix.append(cur)
        idx.append(ix)
        reps.append(stat([xv[i] for i in ix]))
    m = _mean(reps)
    se = math.sqrt(ssum((v - m) ** 2 for v in reps) / (nboot - 1))
    return RichResult(
        payload={
            "replicates": reps,
            "se": se,
            "ci": [_q7(reps, alpha / 2), _q7(reps, 1 - alpha / 2)],
            "indices": idx,
            "estimate": stat(xv),
        }
    )


def bootstrap_bands(replicates, estimate=None, *, alpha: float = 0.05) -> RichResult:
    r"""Pointwise percentile and simultaneous (max-``|t|``) bootstrap confidence bands for a field or curve.

    ``replicates`` is ``R x m`` (one bootstrap field per row). Pointwise:
    the ``alpha/2`` and ``1 - alpha/2`` quantiles at each location.
    Simultaneous: ``estimate +- c s`` with ``s`` the bootstrap standard
    deviations and ``c`` the ``1 - alpha`` quantile of ``max_j |theta*_j -
    estimate_j| / s_j`` over replicates (Davison and Hinkley 1997,
    section 5.2; Wasserman 2006), so the whole field is covered with
    probability ``1 - alpha``.

    References
    ----------
    Davison, A. C. and Hinkley, D. V. (1997). *Bootstrap Methods and their
    Application*. Cambridge University Press.

    Examples
    --------
    >>> r = bootstrap_bands([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]])
    >>> r.estimate
    [1.0, 2.0]
    """
    R = [[float(v) for v in row] for row in replicates]
    n, m = len(R), len(R[0])
    est = [_mean([r[j] for r in R]) for j in range(m)] if estimate is None else [float(v) for v in estimate]
    sd = [math.sqrt(ssum((r[j] - _mean([q[j] for q in R])) ** 2 for r in R) / (n - 1)) for j in range(m)]
    lo = [_q7([r[j] for r in R], alpha / 2) for j in range(m)]
    hi = [_q7([r[j] for r in R], 1 - alpha / 2) for j in range(m)]
    tmax = [
        max(abs(r[j] - est[j]) / sd[j] for j in range(m) if sd[j] > 0) if any(s > 0 for s in sd) else 0.0 for r in R
    ]
    c = _q7(tmax, 1 - alpha)
    return RichResult(
        payload={
            "estimate": est,
            "pointwise_lower": lo,
            "pointwise_upper": hi,
            "simultaneous_lower": [e - c * s for e, s in zip(est, sd)],
            "simultaneous_upper": [e + c * s for e, s in zip(est, sd)],
            "critical_value": c,
        }
    )


def toroidal_shift_test(a, b, *, exact: bool = True, nshift: int = 199, seed: int = 1) -> RichResult:
    r"""Test of association between two gridded maps by toroidal shifts of the second (Upton and Fingleton 1985).

    The statistic is the Pearson correlation of the maps. Shifting ``b``
    cyclically by ``(di, dj)`` keeps both maps' autocorrelation (unlike a
    permutation), giving the null distribution: all ``nrow ncol - 1``
    non-zero shifts when ``exact`` (deterministic) or ``nshift`` random
    shifts (Philox stream 0). Two-sided p-value ``(1 + #{|r_s| >= |r|}) / (1
    + S)``.

    References
    ----------
    Upton, G. J. G. and Fingleton, B. (1985). *Spatial Data Analysis by
    Example*, Vol. 1. Wiley.
    Fortin, M.-J. and Dale, M. R. T. (2005). *Spatial Analysis: A Guide for
    Ecologists*. Cambridge University Press.

    Examples
    --------
    >>> r = toroidal_shift_test([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    >>> round(r.statistic, 12), r.n_shifts
    (1.0, 3)
    """
    A = [[float(v) for v in row] for row in a]
    B = [[float(v) for v in row] for row in b]
    nr, nc = len(A), len(A[0])
    fa = [v for row in A for v in row]

    def cor(bb):
        fb = [v for row in bb for v in row]
        ma, mb = _mean(fa), _mean(fb)
        return ssum((x - ma) * (y - mb) for x, y in zip(fa, fb)) / math.sqrt(
            ssum((x - ma) ** 2 for x in fa) * ssum((y - mb) ** 2 for y in fb)
        )

    def shift(di, dj):
        return [[B[(i + di) % nr][(j + dj) % nc] for j in range(nc)] for i in range(nr)]

    r0 = cor(B)
    if exact:
        shifts = [(di, dj) for di in range(nr) for dj in range(nc) if di or dj]
    else:
        u = [float(v) for v in random_uniform(2 * nshift, seed=seed, stream=0)]
        shifts = [(min(int(u[2 * k] * nr), nr - 1), min(int(u[2 * k + 1] * nc), nc - 1)) for k in range(nshift)]
    sims = [cor(shift(di, dj)) for di, dj in shifts]
    return RichResult(
        payload={
            "statistic": r0,
            "simulated": sims,
            "n_shifts": len(sims),
            "p_value": (1 + sum(1 for v in sims if abs(v) >= abs(r0) - 1e-12)) / (1 + len(sims)),
        }
    )


def cheatsheet() -> str:
    return "block_bootstrap_grid / stationary_bootstrap / bootstrap_bands / toroidal_shift_test -> spatial resampling."
