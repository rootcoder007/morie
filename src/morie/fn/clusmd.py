# morie.fn -- function file (rootcoder007/morie)
r"""Butina clustering: exclusion spheres over a similarity threshold.

**The method.** Count, for every compound, how many others lie within
a Tanimoto threshold of it. Take the compound with the most such
neighbours as a cluster centre; it and all of its neighbours become a
cluster and are removed from the pool. Repeat. Whatever is left with
no neighbours becomes a singleton.

The result is not a partition into "natural" groups -- it is a set of
**exclusion spheres**. Every member of a cluster is within the
threshold *of the centroid*, but two members need not be within the
threshold of each other, and the centroids are guaranteed to be
mutually dissimilar. That is a much weaker and much more honest claim
than hierarchical clustering makes, and it is the reason the method
is O(N^2) rather than O(N^3) and runs on databases.

**One decision the paper leaves open.** After removing a cluster, the
neighbour counts of the survivors are stale -- some of their
neighbours are gone. ``recount=False`` keeps the original counts
(the exclusion-sphere reading, and what makes the method a single
pass); ``recount=True`` recomputes them each round, which costs more
and can give different, usually more balanced, clusters. Both are
provided because the choice changes the answer and should be visible.

**The threshold is the model.** There is no objective function being
optimised here and no number of clusters to choose: the threshold
alone determines the outcome. At 1.0 only identical fingerprints
cluster; at 0.0 everything is one cluster. Reporting the threshold
alongside the clusters is not decoration.

References
----------
Butina, D. (1999) "Unsupervised data base clustering based on
Daylight's fingerprint and Tanimoto similarity: a fast and automated
way to cluster small and large data sets", *Journal of Chemical
Information and Computer Sciences* 39(4), 747-750,
doi:10.1021/ci9803381. The neighbour-count ordering, the exclusion
sphere around each selected centroid, the singleton handling, and the
0.8 Tanimoto working threshold reproduced as the default.

Willett, P., Barnard, J. M. & Downs, G. M. (1998) "Chemical
similarity searching", *Journal of Chemical Information and Computer
Sciences* 38(6), 983-996, doi:10.1021/ci9800211, for the Tanimoto
coefficient itself; see :mod:`morie.fn.sasimi`.
"""

from ._richresult import RichResult
from .sasimi import fingerprint, tanimoto

__all__ = ["neighbour_lists", "butina_clusters", "cluster_summary",
           "butina_clustering"]


def neighbour_lists(fps, threshold=0.8):
    r"""For each compound, the set of others within ``threshold``."""
    th = float(threshold)
    if not 0.0 <= th <= 1.0:
        raise ValueError("clusmd: the threshold must lie in [0, 1], "
                         "got %g" % th)
    F = [fingerprint(x) for x in fps]
    if len(F) < 1:
        raise ValueError("clusmd: no compounds given")
    n = len(F)
    nb = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if tanimoto(F[i], F[j]) >= th:
                nb[i].add(j)
                nb[j].add(i)
    return nb


def butina_clusters(fps, threshold=0.8, recount=False):
    r"""The exclusion-sphere clusters, largest sphere first."""
    nb = neighbour_lists(fps, threshold)
    n = len(nb)
    live = set(range(n))
    clusters = []
    while live:
        if recount:
            counts = {i: len(nb[i] & live) for i in live}
        else:
            counts = {i: len(nb[i]) for i in live}
        # Ties broken by index, so the result is deterministic.
        centre = min(live, key=lambda i: (-counts[i], i))
        members = sorted({centre} | (nb[centre] & live))
        clusters.append({"centroid": centre, "members": members,
                         "size": len(members)})
        live -= set(members)
    clusters.sort(key=lambda c: (-c["size"], c["centroid"]))
    return clusters


def cluster_summary(clusters):
    r"""Sizes, singleton count, and the assignment as a flat list."""
    n = sum(c["size"] for c in clusters)
    assign = [None] * n
    for k, c in enumerate(clusters):
        for m in c["members"]:
            assign[m] = k
    return {"n_clusters": len(clusters), "n_compounds": n,
            "sizes": [c["size"] for c in clusters],
            "n_singletons": sum(1 for c in clusters
                                if c["size"] == 1),
            "assignment": assign,
            "centroids": [c["centroid"] for c in clusters]}


def butina_clustering(fps, threshold=0.8, recount=False):
    r"""Entry point: cluster fingerprints by exclusion sphere."""
    cl = butina_clusters(fps, threshold, recount)
    s = cluster_summary(cl)
    return RichResult(payload={
        "estimate": cl, "clusters": cl, "threshold": float(threshold),
        "recount": bool(recount),
        "n_clusters": s["n_clusters"], "sizes": s["sizes"],
        "n_singletons": s["n_singletons"],
        "assignment": s["assignment"], "centroids": s["centroids"],
        "method": "Butina (1999) exclusion-sphere clustering at "
                  "Tanimoto >= %g" % float(threshold),
    })


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
butina_cluster = butina_clusters
butinacluster = butina_clusters
