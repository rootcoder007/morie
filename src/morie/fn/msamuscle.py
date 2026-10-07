# morie.fn -- function file (rootcoder007/morie)
"""MUSCLE-style multiple sequence alignment (Edgar 2004): k-mer distances and a UPGMA guide tree,
progressive profile-profile alignment, a second tree from Kimura distances of the draft, and
tree-dependent restricted partitioning refinement of the sum-of-pairs score."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["muscle_align", "sum_of_pairs_score"]

# NCBI BLOSUM62 (Henikoff and Henikoff 1992), ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62
_B62_ALPHA = "ARNDCQEGHILKMFPSTWYVBZX*"
_B62_ROWS = """
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0 -2 -1 0 -4
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3 -1 0 -1 -4
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3 3 0 -1 -4
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3 4 1 -1 -4
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2 0 3 -1 -4
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2 1 4 -1 -4
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3 -1 -2 -1 -4
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3 0 0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1 -4 -3 -1 -4
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2 0 1 -1 -4
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2 -2 -1 -2 -4
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2 0 0 0 -4
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0 -1 -1 0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1 -3 -2 -1 -4
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4 -3 -2 -1 -4
-2 -1 3 4 -3 0 1 -1 0 -3 -4 0 -3 -3 -2 0 -1 -4 -3 -3 4 1 -1 -4
-1 0 0 1 -3 3 4 -2 0 -3 -3 1 -1 -3 -1 0 -1 -3 -2 -2 1 4 -1 -4
0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 0 0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 1
"""
_NUC = "ACGTUN"


def _matrix(seqs, matrix):
    letters = sorted({ch for s in seqs for ch in s})
    if isinstance(matrix, dict):
        return letters, {
            (a, b): float(matrix[(a, b)] if (a, b) in matrix else matrix[(b, a)]) for a in letters for b in letters
        }
    if matrix is None:
        matrix = "nucleotide" if all(ch in _NUC for ch in letters) else "blosum62"
    if matrix == "blosum62":
        rows = [[float(v) for v in line.split()] for line in _B62_ROWS.strip().splitlines()]
        idx = {ch: i for i, ch in enumerate(_B62_ALPHA)}
        bad = [ch for ch in letters if ch not in idx]
        if bad:
            raise ValueError("letters not in BLOSUM62: " + "".join(bad))
        return letters, {(a, b): rows[idx[a]][idx[b]] for a in letters for b in letters}
    if matrix == "nucleotide":
        return letters, {
            (a, b): (-2.0 if "N" in (a, b) else (5.0 if a == b or {a, b} == {"T", "U"} else -4.0))
            for a in letters
            for b in letters
        }
    raise ValueError("matrix must be 'blosum62', 'nucleotide' or a dict")


def _kmer_distance(x, y, k):
    cx = {}
    cy = {}
    for i in range(len(x) - k + 1):
        cx[x[i : i + k]] = cx.get(x[i : i + k], 0) + 1
    for i in range(len(y) - k + 1):
        cy[y[i : i + k]] = cy.get(y[i : i + k], 0) + 1
    den = min(len(x), len(y)) - k + 1
    if den <= 0:
        return 1.0
    common = 0
    for t in sorted(cx):
        if t in cy:
            common += min(cx[t], cy[t])
    return 1.0 - common / den


def _upgma(D):
    """UPGMA; returns merges [(left, right, height)] with new node ids n, n+1, ... and the node children."""
    n = len(D)
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i][j]
    size = {i: 1 for i in range(n)}
    active = list(range(n))
    merges = []
    nxt = n
    while len(active) > 1:
        best = None
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                i, j = active[a], active[b]
                d = dist[(i, j)]
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        merges.append((i, j, d / 2.0))
        active.remove(i)
        active.remove(j)
        for m in active:
            dim = dist[(min(i, m), max(i, m))]
            djm = dist[(min(j, m), max(j, m))]
            dist[(m, nxt)] = (size[i] * dim + size[j] * djm) / (size[i] + size[j])
        size[nxt] = size[i] + size[j]
        active.append(nxt)
        nxt += 1
    return merges


def _profile(rows, letters):
    """Column frequency vectors (gaps excluded, divided by the number of sequences)."""
    ns = len(rows)
    cols = []
    for c in range(len(rows[0])):
        f = [0.0] * len(letters)
        for r in rows:
            ch = r[c]
            if ch != "-":
                f[letters.index(ch)] += 1.0
        cols.append([v / ns for v in f])
    return cols


def _align_profiles(A, B, letters, S, go, ge):
    """Gotoh alignment of two profiles (lists of aligned strings); returns the merged rows A + B."""
    pa = _profile(A, letters)
    pb = _profile(B, letters)
    L = len(letters)
    M = [[S[(letters[a], letters[b])] for b in range(L)] for a in range(L)]
    m, n = len(pa), len(pb)
    pam = []
    for col in pa:
        v = [0.0] * L
        for a in range(L):
            if col[a] != 0.0:
                for b in range(L):
                    v[b] += col[a] * M[a][b]
        pam.append(v)
    ninf = -math.inf
    H = [[ninf] * (n + 1) for _ in range(m + 1)]
    X = [[ninf] * (n + 1) for _ in range(m + 1)]
    Y = [[ninf] * (n + 1) for _ in range(m + 1)]
    H[0][0] = 0.0
    for i in range(1, m + 1):
        X[i][0] = -go - (i - 1) * ge
    for j in range(1, n + 1):
        Y[0][j] = -go - (j - 1) * ge
    for i in range(1, m + 1):
        va = pam[i - 1]
        for j in range(1, n + 1):
            s = 0.0
            cb = pb[j - 1]
            for b in range(L):
                s += va[b] * cb[b]
            H[i][j] = s + max(H[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            X[i][j] = max(H[i - 1][j] - go, X[i - 1][j] - ge, Y[i - 1][j] - go)
            Y[i][j] = max(H[i][j - 1] - go, Y[i][j - 1] - ge, X[i][j - 1] - go)
    i, j = m, n
    state = 0
    best = H[m][n]
    if X[m][n] > best:
        state, best = 1, X[m][n]
    if Y[m][n] > best:
        state = 2
    ops = []
    while i > 0 or j > 0:
        if state == 0:
            ops.append(0)
            prev = (H[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            i -= 1
            j -= 1
        elif state == 1:
            ops.append(1)
            prev = (H[i - 1][j] - go, X[i - 1][j] - ge, Y[i - 1][j] - go)
            i -= 1
        else:
            ops.append(2)
            prev = (H[i][j - 1] - go, X[i][j - 1] - ge, Y[i][j - 1] - go)
            j -= 1
        if i == 0 and j == 0:
            break
        top = max(prev)
        state = prev.index(top)
    ops.reverse()
    outA = ["" for _ in A]
    outB = ["" for _ in B]
    ia = ib = 0
    for op in ops:
        for k, r in enumerate(A):
            outA[k] += r[ia] if op != 2 else "-"
        for k, r in enumerate(B):
            outB[k] += r[ib] if op != 1 else "-"
        if op != 2:
            ia += 1
        if op != 1:
            ib += 1
    return outA + outB


def _pair_score(s, t, S, go, ge):
    score = 0.0
    prev = 0
    for a, b in zip(s, t):
        if a == "-" and b == "-":
            continue
        if a != "-" and b != "-":
            score += S[(a, b)]
            prev = 0
        elif a == "-":
            score -= ge if prev == 1 else go
            prev = 1
        else:
            score -= ge if prev == 2 else go
            prev = 2
    return score


def _sp(rows, S, go, ge):
    total = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            total += _pair_score(rows[i], rows[j], S, go, ge)
    return total


def _progressive(seqs, merges, letters, S, go, ge):
    n = len(seqs)
    groups = {i: ([i], [seqs[i]]) for i in range(n)}
    nxt = n
    for a, b, _h in merges:
        ia, ra = groups.pop(a)
        ib, rb = groups.pop(b)
        groups[nxt] = (ia + ib, _align_profiles(ra, rb, letters, S, go, ge))
        nxt += 1
    ids, rows = groups[nxt - 1]
    out = [""] * n
    for k, i in enumerate(ids):
        out[i] = rows[k]
    return out


def _strip(rows):
    keep = [c for c in range(len(rows[0])) if any(r[c] != "-" for r in rows)]
    return ["".join(r[c] for c in keep) for r in rows]


def sum_of_pairs_score(alignment, matrix=None, gap_open: float = 10.0, gap_extend: float = 1.0) -> float:
    r"""Sum-of-pairs score of an alignment with affine gaps on each pairwise projection.

    For every pair of rows, columns gapped in both are dropped; aligned
    residues score ``S(a, b)``, and a gap run of length ``L`` in either row
    costs ``gap_open + (L - 1) gap_extend``.

    Examples
    --------
    >>> sum_of_pairs_score(["AC-GT", "ACGGT"], matrix="nucleotide")
    10.0
    """
    _letters, S = _matrix([r.replace("-", "") for r in alignment], matrix)
    return _sp(list(alignment), S, gap_open, gap_extend)


def muscle_align(
    sequences, matrix=None, gap_open: float = 10.0, gap_extend: float = 1.0, k: int = 3, max_iters: int = 16
) -> RichResult:
    r"""MUSCLE-style progressive multiple alignment with tree-dependent refinement (Edgar 2004).

    1. Draft: k-mer distance ``1 - F``, ``F = sum_t min(n_x(t), n_y(t)) /
       (min(|x|, |y|) - k + 1)``, a UPGMA guide tree, and progressive
       profile-profile alignment (Gotoh affine gaps; the column score is the
       frequency-weighted mean substitution score).
    2. Improved: Kimura distances ``-ln(1 - p - p^2/5)`` (``p`` the
       fraction of differing residues in the draft; 10 when undefined), a
       second UPGMA tree and a new progressive alignment.
    3. Refinement: edges of the second tree, deepest first, split the
       sequences in two; the two sub-alignments are re-aligned and the result
       kept when the sum-of-pairs score rises; passes repeat until none helps
       or ``max_iters``.

    Simplifications against MUSCLE 3: no sequence weighting, constant gap
    penalties (also at the ends), and the BLOSUM62 or +5/-4 nucleotide score
    in place of the VTML/log-expectation profile function.

    Parameters
    ----------
    sequences : list of str
        Unaligned sequences (upper case).
    matrix : str or dict, optional
        ``"blosum62"``, ``"nucleotide"`` or a ``{(a, b): score}`` dict;
        chosen from the alphabet when omitted.
    gap_open, gap_extend : float
        Affine gap costs (a gap of length ``L`` costs ``gap_open + (L - 1)
        gap_extend``).
    k : int
        k-mer length for the draft distances.
    max_iters : int
        Maximum refinement passes.

    Returns
    -------
    RichResult
        ``alignment`` (rows in input order), ``sp_score``, ``sp_draft``,
        ``sp_improved``, ``accepted`` (refinement moves kept), ``tree1``,
        ``tree2`` (UPGMA merges ``(left, right, height)``).

    References
    ----------
    Edgar, R. C. (2004). MUSCLE: multiple sequence alignment with high
    accuracy and high throughput. *Nucleic Acids Research*, 32(5),
    1792-1797.
    Edgar, R. C. (2004). MUSCLE: a multiple sequence alignment method with
    reduced time and space complexity. *BMC Bioinformatics*, 5, 113.
    Kimura, M. (1983). *The Neutral Theory of Molecular Evolution*.
    Cambridge University Press.

    Examples
    --------
    >>> r = muscle_align(["ACGTACGT", "ACGACGT", "ACGTTACGT"])
    >>> [s.replace("-", "") for s in r.alignment] == ["ACGTACGT", "ACGACGT", "ACGTTACGT"]
    True
    >>> len({len(s) for s in r.alignment})
    1
    """
    seqs = [s.upper() for s in sequences]
    if len(seqs) < 2:
        raise ValueError("need at least two sequences")
    letters, S = _matrix(seqs, matrix)
    n = len(seqs)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = _kmer_distance(seqs[i], seqs[j], k)
    tree1 = _upgma(D)
    draft = _progressive(seqs, tree1, letters, S, gap_open, gap_extend)
    sp_draft = _sp(draft, S, gap_open, gap_extend)
    for i in range(n):
        for j in range(i + 1, n):
            same = tot = 0
            for a, b in zip(draft[i], draft[j]):
                if a != "-" and b != "-":
                    tot += 1
                    same += a == b
            p = 1.0 - same / tot if tot else 1.0
            arg = 1.0 - p - p * p / 5.0
            D[i][j] = D[j][i] = -math.log(arg) if arg > 0 else 10.0
    tree2 = _upgma(D)
    aln = _progressive(seqs, tree2, letters, S, gap_open, gap_extend)
    sp = _sp(aln, S, gap_open, gap_extend)
    sp_improved = sp
    # refinement: node ids 0..2n-2, children from the merges; root = 2n-2
    kids = {n + t: (a, b) for t, (a, b, _h) in enumerate(tree2)}
    depth = {2 * n - 2: 0}
    for node in range(2 * n - 2, n - 1, -1):
        for c in kids[node]:
            depth[c] = depth[node] + 1

    def leaves(node):
        return [node] if node < n else leaves(kids[node][0]) + leaves(kids[node][1])

    edges = sorted(range(2 * n - 2), key=lambda v: (-depth[v], v))
    accepted = 0
    for _ in range(max_iters):
        improved = False
        for v in edges:
            inside = sorted(leaves(v))
            outside = [i for i in range(n) if i not in inside]
            if not outside:
                continue
            A = _strip([aln[i] for i in inside])
            B = _strip([aln[i] for i in outside])
            merged = _align_profiles(A, B, letters, S, gap_open, gap_extend)
            cand = [""] * n
            for k2, i in enumerate(inside + outside):
                cand[i] = merged[k2]
            sc = _sp(cand, S, gap_open, gap_extend)
            if sc > sp:
                aln, sp = cand, sc
                accepted += 1
                improved = True
        if not improved:
            break
    return RichResult(
        payload={
            "alignment": aln,
            "sp_score": sp,
            "sp_draft": sp_draft,
            "sp_improved": sp_improved,
            "accepted": accepted,
            "tree1": tree1,
            "tree2": tree2,
        }
    )


def cheatsheet() -> str:
    return "muscle_align(sequences) -> MUSCLE-style progressive MSA with refinement; sum_of_pairs_score(alignment)."


# alias kept from the retired placeholder of the same name
muscle_msa = muscle_align
