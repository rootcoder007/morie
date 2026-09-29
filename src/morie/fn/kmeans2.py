# morie.fn -- function file (rootcoder007/morie)
"""K-means clustering with R-style verbose result."""

from ._richresult import RichResult
from ._rng import random_uniform
from .clusops import kmeans_lloyd


def kmeans2(X, n_clusters: int = 3, n_init: int = 10, random_state: int = 42):
    r"""K-means (Lloyd's algorithm) with ``n_init`` random starts, keeping the smallest within-cluster SSE.

    Start ``s`` takes ``k`` distinct observations as centres, drawn by a
    partial Fisher-Yates shuffle driven by Philox stream ``s`` of
    ``random_state`` (the R arm draws the same starts); each start is run
    to convergence by :func:`morie.fn.clusops.kmeans_lloyd` (Lloyd 1982;
    ``stats::kmeans(algorithm = "Lloyd")``) and the solution with the least
    total within-cluster sum of squares is returned (Hartigan and Wong 1979
    discuss the need for several starts). ``labels`` are 0-based.

    References
    ----------
    Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE
    Transactions on Information Theory* 28, 129-137.

    Examples
    --------
    >>> r = kmeans2([[0, 0], [0, 1], [5, 5], [5, 6], [9, 0], [9, 1]], n_clusters=3)
    >>> r["inertia"]
    1.5
    """
    P = [[float(v) for v in r] for r in (X.tolist() if hasattr(X, "tolist") else X)]
    n = len(P)
    k = int(n_clusters)
    if k < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
    best = None
    for s in range(int(n_init)):
        u = random_uniform(n, seed=random_state, stream=s)
        idx = list(range(n))
        for i in range(k):
            j = i + int(float(u[i]) * (n - i))
            idx[i], idx[j] = idx[j], idx[i]
        r = kmeans_lloyd(P, [P[i] for i in idx[:k]])
        if best is None or r["tot_withinss"] < best["tot_withinss"]:
            best = r
    labels = [c - 1 for c in best["cluster"]]
    sizes = [labels.count(i) for i in range(k)]
    cent = [list(c) for c in best["centers"]]
    return RichResult(
        title="K-means clustering",
        summary_lines=[
            ("k (clusters)", k),
            ("n observations", n),
            ("Inertia (within-cluster SSE)", best["tot_withinss"]),
            ("n_init restarts", n_init),
        ],
        tables=[
            {
                "title": "Cluster summary:",
                "headers": ["Cluster", "Size", "Centroid"],
                "rows": [[f"Cluster {i}", sizes[i], ", ".join(f"{c:.3g}" for c in cent[i])] for i in range(k)],
            }
        ],
        warnings=[] if min(sizes) > 1 else ["one or more clusters has only 1 point - check k or initialization."],
        payload={"labels": labels, "centroids": cent, "inertia": best["tot_withinss"], "sizes": sizes},
    )


def cheatsheet() -> str:
    return "kmeans2: kmeans2(X, n_clusters, n_init, random_state) -> Lloyd k-means, best of n_init Philox starts."
