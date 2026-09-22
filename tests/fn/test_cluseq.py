"""Tests for cluseq.sequence_clustering."""

from morie.fn import _array_core as np

from morie.fn.cluseq import sequence_clustering


def _hamming(a, b):
    """Reference pairwise Hamming distance for two equal-length strings."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _pairwise_hamming(seqs):
    """Reference NxN Hamming distance matrix (lists-of-lists)."""
    n = len(seqs)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _hamming(seqs[i], seqs[j])
            D[i][j] = d
            D[j][i] = d
    return D


def _reference_clustering(sequences, snp_threshold):
    """Reference implementation of the documented single-linkage formula."""
    seqs = list(sequences)
    n = len(seqs)
    thr = int(snp_threshold)
    D = _pairwise_hamming(seqs)

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
    return {
        "estimate": K,
        "z": z,
        "counts": counts,
        "n_clusters": K,
        "distances": D,
        "max_distance": float(mx),
        "n": n,
    }


def test_cluseq_basic():
    """Test basic functionality against the documented formula."""
    sequences = [
        "ACGTACGT",
        "ACGTACGA",
        "ACGTACGG",
        "TTTTTTTT",
    ]
    snp_threshold = 5

    result = sequence_clustering(sequences, snp_threshold)

    # The return type is documented as a dict.
    assert isinstance(result, dict)

    # The docstring lists the keys the function returns.
    for key in (
        "estimate", "z", "counts", "n_clusters",
        "distances", "max_distance", "n",
    ):
        assert key in result, f"missing documented key: {key}"

    # Compute expected values independently from the formula, on the same inputs.
    expected = _reference_clustering(sequences, snp_threshold)

    assert result["n"] == expected["n"] == 4
    assert result["estimate"] == expected["estimate"]
    assert result["n_clusters"] == expected["n_clusters"]
    assert result["z"] == expected["z"]
    assert result["counts"] == expected["counts"]
    assert result["distances"] == expected["distances"]
    assert result["max_distance"] == expected["max_distance"]

    # Sanity checks against the input geometry and the pairwise distance matrix.
    assert len(result["z"]) == 4
    assert len(result["counts"]) == result["n_clusters"]
    assert sum(result["counts"]) == 4
    assert len(result["distances"]) == 4
    for row in result["distances"]:
        assert len(row) == 4
    # Diagonal of the pairwise distance matrix is zero.
    for i in range(4):
        assert result["distances"][i][i] == 0

    # Documented behaviour: estimate equals n_clusters for this routine.
    assert result["estimate"] == result["n_clusters"]


def test_cluseq_edge():
    """Test edge cases with documented shapes: equal-length sequences and int threshold."""
    sequences = [
        "ACGT",
        "ACGT",
        "ACGT",
    ]
    snp_threshold = 0

    result = sequence_clustering(sequences, snp_threshold)

    assert isinstance(result, dict)

    expected = _reference_clustering(sequences, snp_threshold)

    # All sequences are identical and the threshold is 0: every distinct
    # sequence is its own cluster, but with identical inputs there is one cluster.
    assert result["n"] == expected["n"] == 3
    assert result["estimate"] == 1
    assert result["n_clusters"] == 1
    assert result["z"] == [0, 0, 0]
    assert result["counts"] == [3]
    assert result["max_distance"] == 0.0
    # Pairwise distance matrix is all zeros in this case.
    for row in result["distances"]:
        assert all(v == 0 for v in row)
