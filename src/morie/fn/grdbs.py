# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DBSCAN core-point predicate: |N_eps(x)| >= min_samples."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dbscan_core_point"]

_METHOD = "DBSCAN core-point identification"


def geron_dbscan_core_point(X, eps, min_samples, metric="euclidean"):
    r"""Which points are dense enough to seed a DBSCAN cluster.

    .. math::
        \text{core}(x) = \mathbf{1}\bigl[\,
        |\{x' : \|x - x'\| \le \varepsilon\}| \ge \text{min\_samples}\,\bigr]

    The neighbourhood **includes the point itself**, matching the usual
    convention -- so ``min_samples=1`` makes every point core, which is
    why implementations start at 2.  Points that are not core but sit
    inside a core point's neighbourhood are *border* points; the rest are
    noise, and DBSCAN is the rare clustering method that is allowed to
    say "none of these".

    Parameters
    ----------
    X : array-like, shape (m, d)
        Points. A 1-D array is treated as one feature.
    eps : float
        Neighbourhood radius, positive.
    min_samples : int
        Density threshold, at least 1.
    metric : {"euclidean", "manhattan", "chebyshev"}, optional
        Distance used.

    Returns
    -------
    RichResult
        Payload keys ``is_core``, ``is_border``, ``is_noise``,
        ``neighbor_counts``, ``neighbors``, ``n_core``, ``n_noise``,
        ``estimate`` (fraction of points that are core), ``n``,
        ``method``.

    References
    ----------
    Géron Ch 8, DBSCAN section.

    Examples
    --------
    Two points close together and one far away, at ``eps=1``,
    ``min_samples=2``:

    >>> r = geron_dbscan_core_point([[0.0], [0.5], [10.0]], eps=1.0, min_samples=2)
    >>> r["neighbor_counts"]
    [2, 2, 1]
    >>> r["is_core"]
    [True, True, False]
    >>> r["is_noise"]
    [False, False, True]

    Raising the threshold above the local density leaves only noise:

    >>> geron_dbscan_core_point([[0.0], [0.5], [10.0]], eps=1.0,
    ...                         min_samples=3)["n_core"]
    0
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"X must be a non-empty 2-D (m, d) array, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a positive finite float, got {eps}.")
    min_samples = int(min_samples)
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}.")

    diff = X[:, None, :] - X[None, :, :]
    if metric == "euclidean":
        D = np.sqrt(np.sum(diff**2, axis=2))
    elif metric == "manhattan":
        D = np.sum(np.abs(diff), axis=2)
    elif metric == "chebyshev":
        D = np.max(np.abs(diff), axis=2)
    else:
        raise ValueError(
            f"metric must be 'euclidean', 'manhattan' or 'chebyshev', got {metric!r}."
        )

    within = D <= eps
    counts = within.sum(axis=1)
    core = counts >= min_samples
    # Border: not core, but reachable from some core point.
    border = (~core) & np.any(within & core[None, :], axis=1)
    noise = (~core) & (~border)
    neighbors = [np.flatnonzero(row).tolist() for row in within]

    return RichResult(
        title="DBSCAN core points",
        summary_lines=[("Core", int(core.sum())), ("Border", int(border.sum())),
                       ("Noise", int(noise.sum()))],
        payload={
            "is_core": core.tolist(),
            "is_border": border.tolist(),
            "is_noise": noise.tolist(),
            "neighbor_counts": counts.tolist(),
            "neighbors": neighbors,
            "n_core": int(core.sum()),
            "n_noise": int(noise.sum()),
            "eps": eps,
            "min_samples": min_samples,
            "estimate": float(core.mean()),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdbs: DBSCAN core point if |N_eps(x)| >= min_samples (self included); border/noise too"
