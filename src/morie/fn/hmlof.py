# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local outlier factor (LOF) using local reachability density."""

import numpy as np

from ._richresult import RichResult
from .hmmds import pairwise_distances

__all__ = ["geron_local_outlier_factor"]

_METHOD = "Local outlier factor"


def geron_local_outlier_factor(X, n_neighbors=20, contamination=None):
    """
    Local outlier factor (LOF) using local reachability density.

    Formula: LOF_k(x) = avg_{y in N_k(x)} lrd_k(y) / lrd_k(x)

    LOF is a *relative* density score, and that is what distinguishes it
    from a global distance threshold: a point in a sparse but genuine
    cluster has a low absolute density and a LOF near 1, because its
    neighbours are equally sparse.  Only a point whose neighbours are
    substantially denser than it scores above 1.

    The construction, in order:

    ``k-distance(p)`` -- the distance to the k-th nearest neighbour.
    ``reach-dist_k(p, o) = max(k-distance(o), d(p, o))`` -- the
    smoothing term.  Using the raw distance instead makes the density
    estimate wildly unstable inside a dense cluster, which is the point
    of the ``max``.
    ``lrd_k(p) = 1 / mean(reach-dist_k(p, o) for o in N_k(p))``.
    ``LOF_k(p) = mean(lrd_k(o) for o in N_k(p)) / lrd_k(p)``.

    Duplicate points give a zero mean reachability and an infinite
    density; that is raised on rather than silently regularised, since
    it means the neighbourhood has collapsed.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_neighbors : int
        Neighbourhood size ``k``, ``1 <= k < m``.
    contamination : float, optional
        Expected outlier fraction in (0, 0.5]; if given, the
        corresponding number of highest-LOF points are flagged.

    Returns
    -------
    result : RichResult
        Keys: lof, lrd, k_distance, neighbors, is_outlier,
        estimate, n, method.

    Examples
    --------
    A tight cluster plus one distant point: the outlier's LOF is far
    above 1, the cluster's is near it.

    >>> X = [[0.0], [0.1], [0.2], [0.3], [0.4], [10.0]]
    >>> r = geron_local_outlier_factor(X, n_neighbors=2)
    >>> bool(r["lof"][-1] > 3)
    True
    >>> bool(np.all(r["lof"][:5] < 2))
    True

    Uniformly spaced points are all equally typical, so every LOF is
    close to 1:

    >>> u = geron_local_outlier_factor([[float(i)] for i in range(8)], n_neighbors=2)
    >>> bool(np.all(np.abs(u["lof"] - 1.0) < 0.5))
    True

    With a contamination rate the top scorer is flagged:

    >>> f = geron_local_outlier_factor(X, n_neighbors=2, contamination=0.2)
    >>> int(np.count_nonzero(f["is_outlier"])), bool(f["is_outlier"][-1])
    (1, True)

    Sparse-but-genuine clusters are not punished: two clusters of very
    different density both score near 1.

    >>> Y = [[0.0], [0.05], [0.1], [50.0], [60.0], [70.0]]
    >>> d = geron_local_outlier_factor(Y, n_neighbors=2)
    >>> bool(np.all(d["lof"] < 2.5))
    True

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_local_outlier_factor: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_local_outlier_factor: X contains non-finite values")
    m = A.shape[0]
    k = int(n_neighbors)
    if not (1 <= k < m):
        raise ValueError(
            f"geron_local_outlier_factor: n_neighbors must lie in 1..{m - 1} "
            f"(a point is not its own neighbour), got {n_neighbors!r}"
        )

    D = pairwise_distances(A)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="mergesort")
    nbrs = order[:, :k]
    kdist = np.take_along_axis(D, nbrs, axis=1)[:, -1]

    lrd = np.empty(m)
    for p in range(m):
        reach = np.maximum(kdist[nbrs[p]], D[p, nbrs[p]])
        mean_reach = float(np.mean(reach))
        if mean_reach == 0:
            raise ValueError(
                f"geron_local_outlier_factor: point {p} coincides with all {k} of its neighbours, "
                f"so its local reachability density is infinite; remove duplicates or lower n_neighbors"
            )
        lrd[p] = 1.0 / mean_reach

    lof = np.asarray([float(np.mean(lrd[nbrs[p]]) / lrd[p]) for p in range(m)])

    is_outlier = np.zeros(m, dtype=bool)
    if contamination is not None:
        c = float(contamination)
        if not (0.0 < c <= 0.5):
            raise ValueError(f"geron_local_outlier_factor: contamination must lie in (0, 0.5], got {contamination!r}")
        n_out = max(1, int(round(c * m)))
        is_outlier[np.argsort(lof)[::-1][:n_out]] = True

    return RichResult(
        title="Local outlier factor",
        summary_lines=[
            ("Points", int(m)),
            ("k", k),
            ("Max LOF", float(np.max(lof))),
            ("Flagged", int(np.count_nonzero(is_outlier))),
        ],
        interpretation=(
            "LOF is relative: about 1 means as dense as the neighbourhood, well above 1 means the "
            "neighbours are much denser. A sparse but genuine cluster still scores near 1."
        ),
        payload={
            "lof": lof,
            "lrd": lrd,
            "k_distance": kdist,
            "neighbors": nbrs,
            "is_outlier": is_outlier,
            "distances": D,
            "estimate": float(np.max(lof)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlof: local outlier factor from reachability-smoothed local densities; ~1 is typical, >1 is an outlier"
