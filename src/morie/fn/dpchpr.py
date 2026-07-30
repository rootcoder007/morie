# morie.fn -- function file (rootcoder007/morie)
"""Differentially private changepoint detection."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_changepoint"]


def dp_changepoint(y, epsilon=1.0, bounds=None, min_segment=5, seed=None):
    r"""Locate a single changepoint privately by the exponential mechanism.

    For each candidate split :math:`\tau` the utility is the reduction in
    within-segment sum of squares,

    .. math::
        u(\tau) = \mathrm{SS}_{\text{total}}
                 - \mathrm{SS}_{[1,\tau]} - \mathrm{SS}_{(\tau,n]},

    and one candidate is drawn with probability proportional to
    :math:`\exp(\varepsilon u / 2\Delta u)`.

    Selecting the location rather than perturbing a test statistic is what
    makes this work. The location is a discrete choice, so the exponential
    mechanism applies directly; adding noise to a CUSUM or likelihood-ratio
    path instead would need the sensitivity of the whole path, which is much
    larger and yields nothing usable.

    The mechanism always returns *a* location, even when the series has no
    changepoint at all -- there is no built-in null. ``utility_ratio``
    compares the selected split's utility to the best available one, and a
    value near the level expected under noise is the signal that nothing real
    was found.

    Parameters
    ----------
    y : array-like
        Series.
    epsilon : float
        Privacy budget.
    bounds : tuple, optional
        ``(low, high)`` clipping bounds chosen independently of the data.
    min_segment : int
        Minimum points either side of the split.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``changepoint``, ``utility``, ``best_utility``, ``utility_ratio``,
        ``probabilities``.

    References
    ----------
    Cummings, R., Krehbiel, S., Mei, Y., Tuo, R., & Zhang, W. (2018).
        Differentially private change-point detection. *NeurIPS 2018*.

    Examples
    --------
    A clear level shift is located near the truth.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.r_[rng.normal(0, 0.3, 100), rng.normal(4, 0.3, 100)]
    >>> r = dp_changepoint(y, epsilon=20.0, bounds=(-3, 8), seed=1)
    >>> bool(abs(r["changepoint"] - 100) < 15)
    True

    On a series with no changepoint the mechanism still returns one, and the
    utility ratio is what reveals it as noise.

    >>> flat = rng.normal(0, 1, 200)
    >>> rf = dp_changepoint(flat, epsilon=20.0, bounds=(-4, 4), seed=1)
    >>> bool(rf["best_utility"] < r["best_utility"] / 10)
    True

    >>> dp_changepoint([1.0, 2.0], epsilon=1.0, min_segment=5)
    Traceback (most recent call last):
        ...
    ValueError: series too short for min_segment=5
    """
    epsilon, _ = check_budget(epsilon)
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = y.size
    min_segment = int(min_segment)
    if n < 2 * min_segment + 1:
        raise ValueError(f"series too short for min_segment={min_segment}")
    warn = []
    if bounds is None:
        lo, hi = float(y.min()), float(y.max())
        warn.append("bounds were taken from the data, which is a non-private query")
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    yc, lo, hi = clip_to_range(y, lo, hi)

    cand = np.arange(min_segment, n - min_segment)
    ss_tot = float(np.sum((yc - yc.mean()) ** 2))
    util = np.empty(cand.size)
    for i, tau in enumerate(cand):
        a, b = yc[:tau], yc[tau:]
        util[i] = ss_tot - np.sum((a - a.mean()) ** 2) - np.sum((b - b.mean()) ** 2)
    sens = (hi - lo) ** 2
    logp = epsilon * util / (2.0 * sens)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    pick = int(rng.choice(cand.size, p=p))
    best = float(util.max())
    return RichResult(
        title="DP changepoint",
        summary_lines=[("epsilon", epsilon), ("n", int(n)),
                       ("changepoint", int(cand[pick]))],
        warnings=warn + ["the mechanism always returns a location; compare "
                         "best_utility against what noise alone would give "
                         "before believing there is a changepoint"],
        payload={
            "changepoint": int(cand[pick]), "utility": float(util[pick]),
            "best_utility": best,
            "utility_ratio": float(util[pick] / best) if best > 0 else float("nan"),
            "candidates": cand, "probabilities": p,
            "epsilon": epsilon, "n": int(n), "method": "dp_changepoint",
        },
    )


def cheatsheet():
    return "dpchpr: selects a LOCATION by exponential mechanism; always returns one, so check best_utility"
