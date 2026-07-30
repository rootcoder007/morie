# morie.fn -- function file (rootcoder007/morie)
"""Histogram-based outlier score -- Goldstein & Dengel (2012)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["hbos"]


def hbos(X, bins=10, mode="static"):
    r"""Score outliers by the product of per-feature histogram densities.

    Each feature gets its own histogram; the score is

    .. math::
        \mathrm{HBOS}(x) = \sum_{j=1}^{d}
            \log \frac{1}{\hat p_j(x_j)},

    the negative log of the assumed-independent density. Linear in ``n`` and
    embarrassingly parallel, which is why it is the usual first pass on large
    data.

    The independence assumption is the whole limitation and is not a
    technicality: HBOS cannot see a point that is unremarkable on every axis
    separately but impossible jointly. On a tight diagonal band, a point off
    the diagonal but inside both marginals scores as normal. When that matters,
    reach for :func:`~morie.fn.lof.local_outlier_factor` or
    :func:`~morie.fn.mcdAnm.mcd_outlier`.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.
    bins : int
        Bins per feature.
    mode : {"static", "dynamic"}
        ``"static"`` uses equal-width bins; ``"dynamic"`` uses equal-frequency
        bins, which is more robust to skew.

    Returns
    -------
    RichResult
        ``score`` (higher is more outlying), ``rank``, ``densities``,
        ``bin_edges``.

    References
    ----------
    Goldstein, M., & Dengel, A. (2012). Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm. *KI-2012*, 59-63.

    Examples
    --------
    A point far out on one axis lands in the extreme upper tail of scores.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 1, (300, 2)), [[9.0, 0.0]]]
    >>> r = hbos(X, bins=12)
    >>> bool(r["rank"][300] <= 3)
    True

    Note it is not *the* top point, and that is the aggregation at work rather
    than a defect: summing over features dilutes a single-feature outlier,
    because sitting at the mode of the other feature contributes a small term.
    On the offending feature alone it is unambiguously the extreme.

    >>> int(np.argmax(hbos(X[:, :1], bins=12)["score"]))
    300

    The documented blind spot: on a tight diagonal, an off-diagonal point is
    invisible because both marginals look ordinary.

    >>> z = rng.normal(0, 1, 400)
    >>> D = np.column_stack([z, z])
    >>> D = np.r_[D, [[2.0, -2.0]]]              # impossible jointly, fine apart
    >>> s = hbos(D, bins=12)["score"]
    >>> bool(s[-1] < np.quantile(s[:-1], 0.99))
    True

    >>> hbos([[1.0]], bins=0)
    Traceback (most recent call last):
        ...
    ValueError: bins must be at least 1
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if mode not in ("static", "dynamic"):
        raise ValueError('mode must be "static" or "dynamic"')

    dens = np.empty((n, d))
    edges_all = []
    for j in range(d):
        col = X[:, j]
        if mode == "dynamic":
            edges = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)))
            if edges.size < 2:
                edges = np.array([col.min() - 0.5, col.max() + 0.5])
        else:
            lo, hi = col.min(), col.max()
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, bins + 1)
        counts, edges = np.histogram(col, bins=edges)
        width = np.diff(edges)
        p = counts / (counts.sum() * np.where(width > 0, width, 1.0))
        idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, p.size - 1)
        dens[:, j] = np.maximum(p[idx], 1e-12)
        edges_all.append(edges)

    score = -np.log(dens).sum(axis=1)
    order = np.argsort(-score)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return RichResult(
        title="HBOS",
        summary_lines=[("n", n), ("d", d), ("bins", bins), ("mode", mode)],
        payload={
            "score": score, "rank": rank, "densities": dens,
            "bin_edges": edges_all, "mode": mode, "bins": bins,
            "method": "hbos",
        },
    )


def cheatsheet():
    return "hbos: per-feature histograms, assumes independence -- blind to joint-only outliers"
