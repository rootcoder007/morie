import math

import pytest

from morie.fn.msamuscle import muscle_align, sum_of_pairs_score

GLOBINS = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF",
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESF",
    "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKF",
    "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
]


def _nuc(a, b):
    return 5.0 if a == b else -4.0


def _pair(s, t, sc, go=10.0, ge=1.0):
    total, prev = 0.0, 0
    for a, b in zip(s, t):
        if a == "-" and b == "-":
            continue
        if a != "-" and b != "-":
            total += sc(a, b)
            prev = 0
        elif a == "-":
            total -= ge if prev == 1 else go
            prev = 1
        else:
            total -= ge if prev == 2 else go
            prev = 2
    return total


def _gotoh(x, y, sc, go=10.0, ge=1.0):
    m, n = len(x), len(y)
    inf = -math.inf
    H = [[inf] * (n + 1) for _ in range(m + 1)]
    X = [[inf] * (n + 1) for _ in range(m + 1)]
    Y = [[inf] * (n + 1) for _ in range(m + 1)]
    H[0][0] = 0.0
    for i in range(1, m + 1):
        X[i][0] = -go - (i - 1) * ge
    for j in range(1, n + 1):
        Y[0][j] = -go - (j - 1) * ge
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            H[i][j] = sc(x[i - 1], y[j - 1]) + max(H[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            X[i][j] = max(H[i - 1][j] - go, X[i - 1][j] - ge, Y[i - 1][j] - go)
            Y[i][j] = max(H[i][j - 1] - go, Y[i][j - 1] - ge, X[i][j - 1] - go)
    return max(H[m][n], X[m][n], Y[m][n])


def test_alignment_is_valid_and_scores_recompute():
    r = muscle_align(GLOBINS)
    assert [s.replace("-", "") for s in r.alignment] == GLOBINS
    assert len({len(s) for s in r.alignment}) == 1
    assert all(any(s[c] != "-" for s in r.alignment) for c in range(len(r.alignment[0])))
    assert r.sp_score == sum_of_pairs_score(r.alignment)
    assert r.sp_score >= r.sp_improved


def test_two_sequences_reach_the_optimal_affine_score():
    x, y = "ACGTTGCAACGTAGC", "ACGTGCAACGTTAGGC"
    r = muscle_align([x, y])
    best = _gotoh(x, y, _nuc)
    assert _pair(r.alignment[0], r.alignment[1], _nuc) == pytest.approx(best, abs=1e-12)
    assert r.sp_score == pytest.approx(best, abs=1e-12)


def test_nucleotide_sum_of_pairs_and_kmer_tree():
    seqs = ["ACGTTGCAACGT", "ACGTGCAACGTT", "ACTTGCAAGT", "AGGTTGCAACGA", "ACGTTGCACGT"]
    r = muscle_align(seqs)
    sp = math.fsum(_pair(r.alignment[i], r.alignment[j], _nuc) for i in range(5) for j in range(i + 1, 5))
    assert r.sp_score == pytest.approx(sp, abs=1e-9)

    def kd(a, b, k=3):
        ca = [a[i : i + k] for i in range(len(a) - k + 1)]
        cb = [b[i : i + k] for i in range(len(b) - k + 1)]
        common = sum(min(ca.count(t), cb.count(t)) for t in set(ca))
        return 1 - common / (min(len(a), len(b)) - k + 1)

    pairs = sorted((kd(seqs[i], seqs[j]), i, j) for i in range(5) for j in range(i + 1, 5))
    assert tuple(r.tree1[0][:2]) == pairs[0][1:]
    assert r.tree1[0][2] == pytest.approx(pairs[0][0] / 2, abs=1e-15)


def test_identical_sequences_and_errors():
    r = muscle_align(["MKTAYIAK"] * 3)
    assert r.alignment == ["MKTAYIAK"] * 3
    diag = {"M": 5, "K": 5, "T": 5, "A": 4, "Y": 7, "I": 4}
    assert r.sp_score == 3 * sum(diag[c] for c in "MKTAYIAK")
    with pytest.raises(ValueError):
        muscle_align(["ACGT"])
    with pytest.raises(ValueError):
        muscle_align(["MKJ", "MKT"])
