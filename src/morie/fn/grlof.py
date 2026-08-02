# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local outlier factor."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_local_outlier_factor"]

_METHOD = "Local outlier factor"


def geron_local_outlier_factor(X, k=5):
    r"""Density of a point relative to the density of its neighbours.

    With ``k``-distance :math:`d_k(x)`, reachability distance
    :math:`\mathrm{rd}_k(x, o) = \max(d_k(o), \|x - o\|)` and local
    reachability density
    :math:`\mathrm{lrd}(x) = 1 / \overline{\mathrm{rd}_k(x, o)}`,

    .. math::
        \mathrm{LOF}(x) = \frac{1}{k}\sum_{o \in N_k(x)}
        \frac{\mathrm{lrd}(o)}{\mathrm{lrd}(x)}

    The ratio is the point: LOF is *local*, so it flags a point that is
    sparse compared with its own neighbourhood even when a denser
    cluster elsewhere would make it look ordinary on a global distance
    threshold.  Values near 1 mean "as dense as my neighbours"; well
    above 1 means outlier.

    The reachability smoothing (``max`` against the neighbour's own
    ``k``-distance) is what stops a tight pair of duplicates from
    reporting infinite density.

    Parameters
    ----------
    X : array-like, shape (m, n)
    k : int, optional
        Neighbourhood size, ``1 <= k <= m - 1``. Default 5.

    Returns
    -------
    RichResult
        Payload keys ``lof``, ``lrd``, ``k_distance``, ``neighbors``,
        ``most_outlying`` (index of the largest LOF), ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 8, Local Outlier Factor section (Breunig et al. 2000).

    Examples
    --------
    Three points in a tight cluster and one far away: the stray point
    scores far above 1, the cluster members near it.

    >>> X = [[0.0], [1.0], [2.0], [50.0]]
    >>> r = geron_local_outlier_factor(X, k=2)
    >>> r["most_outlying"]
    3
    >>> r["lof"][3] > 5
    True
    >>> max(r["lof"][:3]) < 2
    True

    On a uniform line LOF stays near 1, but not exactly at it: the
    endpoints have a larger ``k``-distance (their second neighbour is
    two steps away), which lowers their ``lrd`` and drags the interior
    ratio to ``(1/1.5) / 1``:

    >>> r2 = geron_local_outlier_factor([[0.0], [1.0], [2.0], [3.0], [4.0]], k=2)
    >>> round(r2["lof"][2], 6)
    0.666667
    >>> [round(v, 6) for v in r2["k_distance"]]
    [2.0, 1.0, 1.0, 1.0, 2.0]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    m = A.shape[0]
    k = int(k)
    if not (1 <= k <= m - 1):
        raise ValueError(
            f"k must lie in [1, {m - 1}] (a point cannot be its own neighbour), got {k}."
        )

    D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    nbrs = order[:, :k]
    kdist = D[np.arange(m), nbrs[:, -1]]
    if np.any(kdist == 0):
        bad = np.flatnonzero(kdist == 0).tolist()
        raise ValueError(
            f"points {bad} have {k} or more exact duplicates, so their local "
            f"reachability density is infinite; deduplicate or lower k."
        )

    reach = np.maximum(kdist[nbrs], D[np.arange(m)[:, None], nbrs])
    lrd = 1.0 / reach.mean(axis=1)
    lof = lrd[nbrs].mean(axis=1) / lrd

    return RichResult(
        title="Local outlier factor",
        summary_lines=[("k", k), ("Max LOF", float(lof.max())),
                       ("Most outlying", int(lof.argmax()))],
        payload={
            "lof": lof.tolist(),
            "lrd": lrd.tolist(),
            "k_distance": kdist.tolist(),
            "neighbors": nbrs.tolist(),
            "most_outlying": int(lof.argmax()),
            "k": k,
            "estimate": lof.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlof: LOF = mean(lrd(neighbour))/lrd(x) with reachability smoothing; ~1 is normal"
