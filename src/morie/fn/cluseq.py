# morie.fn -- function file (rootcoder007/morie)
"""SNP-distance single-linkage sequence clustering."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["sequence_clustering"]


def hamming(a, b):
    if len(a) != len(b):
        raise ValueError("sequences must be the same length to compare")
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def sequence_clustering(sequences, snp_threshold=5):
    """
    SNP-distance sequence clustering

    Formula: SNP-distance + linkage threshold

    Pairwise SNP distances are computed on the aligned sequences and
    isolates are joined into a cluster whenever ANY pair is within the
    threshold -- single linkage, which is what makes the result a
    transitive closure rather than a set of cliques.  A threshold of
    zero leaves every distinct sequence on its own; a threshold at least
    as large as the diameter collapses everything into one cluster.

    Parameters
    ----------
    sequences : sequence
        Aligned sequences of equal length (strings or lists).
    snp_threshold : int
        Maximum SNP distance for a link.

    Returns
    -------
    result : dict
        Keys: estimate (number of clusters), z, counts, n_clusters,
        distances, max_distance, n.

    References
    ----------
    Croucher et al. (2015), Rapid phylogenetic analysis of large samples
    of recombinant bacterial whole genome sequences using Gubbins,
    Nucleic Acids Research 43(3):e15.
    """
    seqs = list(sequences)
    n = len(seqs)
    if n == 0:
        raise ValueError("empty input: no sequences supplied")
    thr = int(snp_threshold)
    if thr < 0:
        raise ValueError("snp_threshold must be non-negative")
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(seqs[i], seqs[j])
            D[i][j] = d
            D[j][i] = d
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] <= thr:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = []
    z = [0] * n
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots.append(r)
        z[i] = roots.index(r)
    K = len(roots)
    counts = [sum(1 for v in z if v == c) for c in range(K)]
    mx = max((D[i][j] for i in range(n) for j in range(i + 1, n)), default=0)
    return RichResult(payload={
        "estimate": K,
        "z": z,
        "counts": counts,
        "n_clusters": K,
        "distances": D,
        "max_distance": float(mx),
        "n": n,
        "method": "SNP-distance single-linkage sequence clustering",
    })


def cheatsheet():
    return "cluseq: SNP-distance sequence clustering"


# compact alias per ledger/NAMING.md
sequenceclustering = sequence_clustering
