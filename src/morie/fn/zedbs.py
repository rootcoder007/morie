"""Spatial DBSCAN cluster detection"""

from ._containers import SpatialResult
from .clusops import dbscan_clusters


def spatial_dbscan(coords, eps=1.0, min_pts=5, *, border_points=True):
    r"""DBSCAN density-based spatial clusters of point locations (Ester et al. 1996).

    Core points have at least ``min_pts`` points (themselves included)
    within Euclidean distance ``eps``; clusters grow from core points in
    index order and border points join the first cluster reaching them
    (label 0 is noise), as ``dbscan::dbscan``. Thin front-end to
    :func:`morie.fn.clusops.dbscan_clusters`; ``statistic`` is the number of
    clusters and ``extra["cluster"]`` the labels.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J. and Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. *KDD-96*, 226-231.

    Examples
    --------
    >>> r = spatial_dbscan([[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9], [5, 5]], eps=1.5, min_pts=3)
    >>> r.statistic, r.extra["cluster"]
    (2, [1, 1, 1, 2, 2, 2, 0])
    """
    lab = list(dbscan_clusters(coords, eps, int(min_pts), border_points=border_points)["cluster"])
    k = max(lab) if lab else 0
    return SpatialResult(
        name="zedbs",
        statistic=k,
        extra={"cluster": lab, "n_noise": lab.count(0), "sizes": [lab.count(c) for c in range(1, k + 1)]},
    )


spat = spatial_dbscan


def cheatsheet() -> str:
    return "spatial_dbscan(coords, eps, min_pts) -> DBSCAN clusters of point locations (Ester et al. 1996)."


# compact alias per ledger/NAMING.md
spatialdbscan = spatial_dbscan
