# morie.fn -- function file (rootcoder007/morie)
"""Empirical-CDF outlier detection -- Li et al. (2022)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["ecod"]


def ecod(X):
    r"""Score outliers from per-feature empirical tail probabilities.

    For each feature compute the left and right empirical CDF tails and
    aggregate their logs:

    .. math::
        O(x) = \max\left(
            -\sum_j \log \hat F_j(x_j),\;
            -\sum_j \log\left(1 - \hat F_j(x_j)\right)
        \right),

    with the choice per feature guided by that feature's skewness, so a
    right-skewed feature is judged on its right tail.

    ECOD has **no hyperparameters at all** -- no ``k``, no bin count, no
    contamination rate -- which is its whole appeal: nothing to tune means
    nothing to tune *wrongly*, and results are reproducible across users.

    It shares HBOS's blind spot: the aggregation is per feature, so a
    joint-only outlier is invisible. What it adds over HBOS is that tails are
    estimated by the ECDF rather than by binning, so there is no bin-width
    choice and extreme values are ranked correctly rather than being lumped
    into an end bin.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.

    Returns
    -------
    RichResult
        ``score``, ``rank``, ``tail_left``, ``tail_right``, ``skewness``.

    References
    ----------
    Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen, G. H. (2022).
        ECOD: Unsupervised outlier detection using empirical cumulative
        distribution functions. *IEEE TKDE*, 35(12), 12181-12193.

    Examples
    --------
    Extremes in either tail land in the top few percent, with nothing to tune.
    Both the high and the low injected point are caught, which is the value of
    aggregating *both* tails rather than one.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 1, (500, 3)), [[8.0, 0.0, 0.0]], [[-8.0, 0.0, 0.0]]]
    >>> r = ecod(X)
    >>> bool(r["rank"][500] < 25 and r["rank"][501] < 25)
    True

    They are not the top two, for the same reason HBOS dilutes: the score sums
    over features, and both points sit at the median of the two features they
    are ordinary in. An outlier in one of ``d`` features competes against
    points mildly extreme in all ``d``.

    >>> bool(np.argsort(-r["score"])[0] < 500)
    True

    Deterministic -- no seed, no hyperparameter, so two runs agree exactly.

    >>> bool(np.array_equal(ecod(X)["score"], ecod(X)["score"]))
    True

    >>> ecod([1.0, 2.0, 3.0])["score"].shape
    (3,)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    n, d = X.shape
    if n < 2:
        raise ValueError("need at least 2 observations")

    left = np.empty((n, d))
    right = np.empty((n, d))
    skew = np.empty(d)
    for j in range(d):
        col = X[:, j]
        ranks = np.searchsorted(np.sort(col), col, side="right")
        left[:, j] = ranks / n
        right[:, j] = (n - np.searchsorted(np.sort(col), col, side="left")) / n
        sd = col.std(ddof=0)
        skew[j] = 0.0 if sd == 0 else float(np.mean(((col - col.mean()) / sd) ** 3))

    lo = -np.log(np.maximum(left, 1.0 / n)).sum(axis=1)
    hi = -np.log(np.maximum(right, 1.0 / n)).sum(axis=1)
    # Per-feature skew decides which tail to trust, then take the stronger.
    auto = -np.log(np.maximum(np.where(skew < 0, left, right), 1.0 / n)).sum(axis=1)
    score = np.maximum(np.maximum(lo, hi), auto)

    order = np.argsort(-score)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return RichResult(
        title="ECOD",
        summary_lines=[("n", n), ("d", d), ("max score", float(score.max()))],
        payload={
            "score": score, "rank": rank, "tail_left": left,
            "tail_right": right, "skewness": skew, "method": "ecod",
        },
    )


def cheatsheet():
    return "ecod: ZERO hyperparameters, ECDF tails; shares HBOS's per-feature blind spot"
