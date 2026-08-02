# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OPTICS: ordering points to identify the clustering structure."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_optics"]


def geron_optics(X, min_samples=5, max_eps=np.inf, eps_cluster=None):
    """
    OPTICS: ordering points to identify clustering structure at multiple densities.

    Formula: reachability plot; clusters from valleys

    DBSCAN needs one eps and therefore one density; OPTICS refuses to
    choose. It walks the points in the order a density-connected sweep
    would visit them and records each one's REACHABILITY -- the smallest
    radius at which it would have joined the cluster it was reached
    from. Plotted in that order the result is a landscape of valleys, one
    per cluster, at whatever density each cluster happens to have; a
    horizontal cut at eps reproduces DBSCAN's answer for that eps
    exactly, which is what ``eps_cluster`` does here.

    Points with infinite reachability are the sweep's restart points.
    They are not noise by themselves; noise is what a given cut leaves
    unlabelled.

    Parameters
    ----------
    X : array-like, shape (n, d)
    min_samples : int, default 5
        Neighbourhood size defining the core distance.
    max_eps : float, default inf
        Largest radius considered; caps the work, not the answer.
    eps_cluster : float, optional
        Cut for extracting labels. Defaults to ``max_eps`` when finite,
        otherwise no labels are produced.

    Returns
    -------
    result : RichResult
        Keys: ordering, reachability, core_distances, labels,
        n_clusters, estimate, n, method.

    Examples
    --------
    Two tight groups ten units apart: within a group the reachability is
    0.1, and the jump between them is the gap.

    >>> X = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
    >>> r = geron_optics(X, min_samples=2, eps_cluster=1.0)
    >>> int(r["n_clusters"])
    2
    >>> sorted(set(int(v) for v in r["labels"]))
    [0, 1]
    >>> round(float(r["core_distances"][0]), 6)
    0.1

    Cutting below the within-group spacing leaves everything unlabelled:

    >>> sorted(set(int(v) for v in geron_optics(X, 2, eps_cluster=0.05)["labels"]))
    [-1]

    References
    ----------
    Geron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"geron_optics: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_optics: X contains non-finite values")
    n = A.shape[0]
    k = int(min_samples)
    if not (1 <= k <= n):
        raise ValueError(f"geron_optics: min_samples must lie in [1, {n}], got {min_samples!r}")
    me = float(max_eps)
    if me <= 0:
        raise ValueError(f"geron_optics: max_eps must be positive, got {max_eps!r}")
    cut = eps_cluster if eps_cluster is not None else (me if np.isfinite(me) else None)
    if cut is not None:
        cut = float(cut)
        if cut <= 0:
            raise ValueError(f"geron_optics: eps_cluster must be positive, got {eps_cluster!r}")

    D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    core = np.full(n, np.inf)
    for i in range(n):
        d = np.sort(D[i])
        if k - 1 < n:
            c = float(d[k - 1])  # k-th nearest INCLUDING the point itself
            if c <= me:
                core[i] = c

    reach = np.full(n, np.inf)
    processed = np.zeros(n, dtype=bool)
    ordering = []
    for start in range(n):
        if processed[start]:
            continue
        seeds = {start: reach[start]}
        while seeds:
            p = min(seeds, key=lambda q: (seeds[q], q))
            del seeds[p]
            if processed[p]:
                continue
            processed[p] = True
            ordering.append(p)
            if not np.isfinite(core[p]):
                continue
            for q in range(n):
                if processed[q] or D[p, q] > me:
                    continue
                nr = max(core[p], float(D[p, q]))
                if nr < reach[q]:
                    reach[q] = nr
                    seeds[q] = nr
                elif q not in seeds and np.isfinite(reach[q]):
                    seeds[q] = reach[q]

    order = np.asarray(ordering, dtype=int)
    labels = np.full(n, -1, dtype=int)
    n_clusters = 0
    if cut is not None:
        cid = -1
        for pos, p in enumerate(order):
            if reach[p] > cut:
                if core[p] <= cut:
                    cid += 1
                    labels[p] = cid
                else:
                    labels[p] = -1
            else:
                labels[p] = cid if cid >= 0 else -1
            del pos
        n_clusters = cid + 1
        # A cluster of a single point at the cut is noise, as in DBSCAN.
        for c in range(n_clusters):
            if int(np.sum(labels == c)) < k:
                labels[labels == c] = -1
        remaining = sorted(set(labels.tolist()) - {-1})
        remap = {old: new for new, old in enumerate(remaining)}
        labels = np.array([remap.get(v, -1) for v in labels], dtype=int)
        n_clusters = len(remaining)

    return RichResult(
        title="OPTICS",
        summary_lines=[("Points", int(n)), ("min_samples", k), ("Clusters at cut", int(n_clusters))],
        interpretation="The reachability plot holds every density at once; a horizontal cut is one DBSCAN run.",
        payload={
            "ordering": order,
            "reachability": reach,
            "reachability_plot": reach[order],
            "core_distances": core,
            "labels": labels,
            "n_clusters": int(n_clusters),
            "eps_cluster": cut,
            "estimate": labels,
            "n": int(n),
            "method": "OPTICS ordering with reachability, cut to labels at eps_cluster",
        },
    )


def cheatsheet():
    return "hmopt: OPTICS reachability ordering and cluster extraction"
