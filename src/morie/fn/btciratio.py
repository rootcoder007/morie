# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap confidence interval for a ratio."""

import numpy as np

from ._richresult import RichResult

__all__ = ["boot_ci_ratio"]


def boot_ci_ratio(x, y, stat_x=None, stat_y=None, B=2000, alpha=0.05,
                  seed=0, paired=False):
    r"""Percentile bootstrap confidence interval for a ratio of two
    statistics, :math:`\theta = T_x(F_x)/T_y(F_y)` (Davison and
    Hinkley 1997, Chs. 2-3 and 5).

    Ratios are exactly where the bootstrap earns its keep and where
    the delta method quietly fails: a ratio's distribution is skewed,
    its Taylor expansion breaks down when the denominator is small,
    and Fieller's classical interval can be a half-line. The
    percentile interval reads the quantiles off the resampled ratios
    directly and needs none of that -- but it inherits the bootstrap
    caveat that a denominator resampling near zero produces wild
    replicates, so the fraction of small-denominator replicates is
    reported, and a large value means the RATIO is the problem, not
    the interval.

    ``paired`` decides the resampling unit, and it is a modelling
    statement: paired data (same subjects in both arms) must be
    resampled as PAIRS to preserve the dependence, while independent
    samples are resampled separately. Getting this wrong biases the
    interval's width in whichever direction the dependence points --
    both modes are tested against each other on correlated data.

    Parameters
    ----------
    x, y : array-like
        The two samples; equal length required when ``paired``.
    stat_x, stat_y : callable, optional
        Statistics for numerator and denominator; the mean when
        omitted.
    B : int, default 2000
        Replicates.
    alpha : float, default 0.05
        Miss probability.
    seed : int, default 0
        Resampling seed.
    paired : bool, default False
        Resample (x_i, y_i) pairs rather than the samples
        independently.

    Returns
    -------
    RichResult
        keys: ``ratio``, ``ci``, ``replicates``, ``se``,
        ``small_denominator_fraction``, ``paired``, ``B``, ``alpha``,
        ``n_x``, ``n_y``, ``method``.

    References
    ----------
    Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
    their Application*, Cambridge University Press, Chs. 2-3, 5.
    Fieller (1954) for the classical interval this sidesteps.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    sx = (lambda d: float(np.mean(d))) if stat_x is None else stat_x
    sy = (lambda d: float(np.mean(d))) if stat_y is None else stat_y
    a = float(alpha)
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {a}.")
    Bn = int(B)
    if Bn < 100:
        raise ValueError(f"need at least 100 replicates for quantiles, "
                         f"got {Bn}.")
    if paired and xv.size != yv.size:
        raise ValueError("paired resampling needs equal-length samples.")
    den0 = float(sy(yv))
    if den0 == 0:
        raise ValueError("the denominator statistic is zero on the data; "
                         "the ratio is undefined.")
    ratio = float(sx(xv)) / den0
    rng = np.random.default_rng(seed)
    reps = np.empty(Bn)
    small = 0
    scale = abs(den0)
    for b in range(Bn):
        if paired:
            idx = rng.integers(0, xv.size, xv.size)
            num = float(sx(xv[idx]))
            den = float(sy(yv[idx]))
        else:
            num = float(sx(xv[rng.integers(0, xv.size, xv.size)]))
            den = float(sy(yv[rng.integers(0, yv.size, yv.size)]))
        if abs(den) < 1e-3 * scale:
            small += 1
        reps[b] = num / den if den != 0 else np.nan
    good = reps[np.isfinite(reps)]
    lo, hi = np.percentile(good, [100 * a / 2, 100 * (1 - a / 2)])
    return RichResult(payload={
        "ratio": ratio, "ci": (float(lo), float(hi)),
        "replicates": reps, "se": float(np.std(good, ddof=1)),
        "small_denominator_fraction": small / Bn,
        "why_bootstrap": "a ratio's distribution is skewed and the delta "
                         "method breaks down for small denominators; the "
                         "percentile interval reads the quantiles directly",
        "paired": bool(paired),
        "pairing_note": "paired data must be resampled as PAIRS to keep the "
                        "dependence; independent samples separately -- this "
                        "is a modelling statement, not a convenience flag",
        "B": Bn, "alpha": a,
        "n_x": int(xv.size), "n_y": int(yv.size),
        "method": "Percentile bootstrap CI for a ratio (Davison-Hinkley 1997)"})


def cheatsheet():
    return "btciratio: percentile CI on resampled ratios -- and paired data resample as pairs"
