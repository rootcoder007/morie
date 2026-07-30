# morie.fn -- function file (rootcoder007/morie)
"""Local outlier factor -- Breunig et al. (2000)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["local_outlier_factor"]


def local_outlier_factor(X, k=20):
    r"""Score outliers by density relative to their own neighbourhood.

    For each point, compare its local reachability density with that of its
    ``k`` nearest neighbours:

    .. math::
        \mathrm{LOF}_k(p) = \frac{1}{|N_k(p)|}
            \sum_{o \in N_k(p)} \frac{\mathrm{lrd}_k(o)}{\mathrm{lrd}_k(p)} .

    Values near 1 mean the point is as dense as its neighbours; substantially
    above 1 means it sits in a sparser region than they do.

    Being *relative* is the point. A global density threshold declares every
    point in a genuinely sparse cluster an outlier; LOF does not, because it
    only ever compares a point to its own neighbourhood. That is what lets it
    work on data with clusters of different densities, which is where
    distance-to-centroid and k-distance methods fail.

    The reachability distance -- :math:`\max(\text{k-dist}(o), d(p,o))` rather
    than the raw distance -- is a smoothing device that stops a single very
    close neighbour from dominating the density estimate.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.
    k : int
        Neighbourhood size, from 1 to ``n - 1``. Too small is unstable; the
        original paper suggests at least 10.

    Returns
    -------
    RichResult
        ``lof`` (score), ``rank``, ``lrd``, ``k_distance``, ``neighbors``.

    References
    ----------
    Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF:
        Identifying density-based local outliers. *SIGMOD 2000*, 93-104.

    Examples
    --------
    A point between two clusters scores well above 1.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 0.3, (100, 2)), rng.normal(6, 0.3, (100, 2)),
    ...           [[3.0, 3.0]]]
    >>> r = local_outlier_factor(X, k=15)
    >>> int(np.argmax(r["lof"]))
    200
    >>> bool(r["lof"][200] > 1.5)
    True

    Inliers sit near 1 -- the scale LOF is designed to make interpretable.

    >>> bool(abs(float(np.median(r["lof"][:200])) - 1.0) < 0.3)
    True

    The reason to use it: a sparse cluster is not flagged wholesale, even
    though its points are far apart in absolute terms.

    >>> Y = np.r_[rng.normal(0, 0.2, (100, 2)), rng.normal(10, 2.0, (100, 2))]
    >>> s = local_outlier_factor(Y, k=15)["lof"]
    >>> bool(float(np.median(s[100:])) < 1.4)
    True

    >>> local_outlier_factor(np.zeros((5, 2)), k=10)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 1 and 4
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be between 1 and {n - 1}")

    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    nbrs = order[:, :k]
    kdist = D[np.arange(n), order[:, k - 1]]

    # Reachability distance smooths away a single very close neighbour.
    reach = np.maximum(kdist[nbrs], D[np.arange(n)[:, None], nbrs])
    lrd = 1.0 / np.maximum(reach.mean(axis=1), 1e-12)
    lof = (lrd[nbrs].mean(axis=1)) / np.maximum(lrd, 1e-12)

    ordr = np.argsort(-lof, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[ordr] = np.arange(n)
    return RichResult(
        title="Local outlier factor",
        summary_lines=[("n", n), ("k", k), ("max LOF", float(lof.max()))],
        payload={
            "lof": lof, "score": lof, "rank": rank, "lrd": lrd,
            "k_distance": kdist, "neighbors": nbrs, "k": k,
            "method": "local_outlier_factor",
        },
    )


def cheatsheet():
    return "lof: density RELATIVE to own neighbourhood, so ~1 is normal; sparse clusters are not flagged"
