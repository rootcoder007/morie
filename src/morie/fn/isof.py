# morie.fn -- function file (rootcoder007/morie)
"""Isolation forest."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["isolation_forest"]


def isolation_forest(X, n_trees=100, sample_size=256, seed=0):
    r"""Score outliers by how few random splits isolate them.

    Each tree splits on a random feature at a random threshold until points
    are separated. Anomalies sit in sparse regions and so are isolated in few
    splits; the score normalises average path length by the expected length
    for a random binary search tree,

    .. math::
        s(x) = 2^{-\,\mathbb{E}[h(x)] / c(n)},
        \qquad c(n) = 2H_{n-1} - 2(n-1)/n,

    giving values near 1 for anomalies and near 0.5 for normal points.

    Isolation forest **looks for sparsity rather than modelling density**,
    which is why it is fast and why it scales to high dimension where
    distance-based scores lose contrast. The cost is that axis-parallel splits
    cannot see structure at an angle: a tight diagonal band is sparse along
    every axis separately, so points inside it score as anomalous. That
    limitation is what the extended isolation forest addresses.

    Subsampling is deliberate, not an approximation. Small samples *improve*
    detection by reducing swamping (normal points in dense regions looking
    anomalous because the tree ran out of them) and masking.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.
    n_trees : int
        Number of trees.
    sample_size : int
        Points per tree; capped at ``n``.
    seed : int
        Seed.

    Returns
    -------
    RichResult
        ``score`` (near 1 is anomalous), ``rank``, ``path_length``,
        ``threshold``.

    References
    ----------
    Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest.
        *ICDM 2008*, 413-422.

    Examples
    --------
    A clear outlier scores highest.

    >>> from morie.fn import _array_core as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 1, (300, 2)), [[12.0, 12.0]]]
    >>> r = isolation_forest(X, n_trees=100, seed=1)
    >>> int(np.argmax(r["score"]))
    300

    Normal points sit near 0.5, the value for an average path length.

    >>> bool(abs(float(np.median(r["score"][:300])) - 0.5) < 0.12)
    True

    Anomalies are isolated in fewer splits, which is the mechanism.

    >>> bool(r["path_length"][300] < float(np.median(r["path_length"][:300])))
    True

    >>> isolation_forest(X, n_trees=0)
    Traceback (most recent call last):
        ...
    ValueError: n_trees must be at least 1
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    n_trees = int(n_trees)
    if n_trees < 1:
        raise ValueError("n_trees must be at least 1")
    psi = int(min(sample_size, n))
    if psi < 2:
        raise ValueError("sample_size must be at least 2")
    rng = np.random.default_rng(seed)
    limit = int(np.ceil(np.log2(psi)))

    def c(m):
        if m <= 1:
            return 0.0
        return 2.0 * (np.log(m - 1) + 0.5772156649) - 2.0 * (m - 1) / m

    def grow(rows, depth):
        """One isolation tree (Liu, Ting & Zhou 2008, iTree): random feature, random split."""
        if depth >= limit or len(rows) <= 1:
            return (len(rows),)
        lo = [min(r[j] for r in rows) for j in range(d)]
        hi = [max(r[j] for r in rows) for j in range(d)]
        wide = [j for j in range(d) if hi[j] > lo[j]]
        if not wide:
            return (len(rows),)
        j = int(rng.choice(wide))
        thr = float(rng.uniform(lo[j], hi[j]))
        return (j, thr,
                grow([r for r in rows if r[j] < thr], depth + 1),
                grow([r for r in rows if r[j] >= thr], depth + 1))

    def path(x, node):
        depth = 0
        while len(node) == 4:
            node = node[2] if x[node[0]] < node[1] else node[3]
            depth += 1
        return depth + c(node[0])

    pts = [[float(v) for v in np.atleast_1d(X[i])] for i in range(n)]
    lengths = np.zeros(n)
    for _ in range(n_trees):
        tree = grow([pts[int(i)] for i in rng.choice(n, psi, replace=False)], 0)
        for i in range(n):
            lengths[i] += path(pts[i], tree)
    lengths /= n_trees
    score = 2.0 ** (-lengths / max(c(psi), 1e-12))
    order = np.argsort(-score, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return RichResult(
        title="Isolation forest",
        summary_lines=[("n", n), ("trees", n_trees), ("sample", psi),
                       ("max score", float(score.max()))],
        warnings=["splits are axis-parallel, so structure at an angle is "
                  "invisible: points inside a tight diagonal band score as "
                  "anomalous"],
        payload={
            "score": score, "rank": rank, "path_length": lengths,
            "threshold": 0.5, "n_trees": n_trees, "sample_size": psi,
            "method": "isolation_forest",
        },
    )


def cheatsheet():
    return "isof: finds SPARSITY not density; subsampling helps (less swamping); blind to diagonal structure"
