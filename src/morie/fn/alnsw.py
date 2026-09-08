# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Smith-Waterman local sequence alignment.

Source: Smith, T. F. and Waterman, M. S. (1981), "Identification of
common molecular subsequences", *Journal of Molecular Biology* 147(1),
195-197, doi:10.1016/0022-2836(81)90087-5 (citation verified against
Crossref).

Two changes to the global recurrence make it local, and both are
essential:

    H(i, 0) = H(0, j) = 0,
    H(i, j) = max{ 0,
                   H(i-1, j-1) + s(a_i, b_j),
                   H(i-1, j) - g,
                   H(i,   j-1) - g },

and the traceback starts at the largest cell anywhere in the matrix and
stops at the first zero, rather than starting at the corner.  The zero
floor is what lets a bad prefix be abandoned instead of dragged along;
without it the matrix is Needleman-Wunsch with a zeroed border and the
answer is a different alignment.  Consequently the score can never be
negative -- the empty alignment always scores zero -- and that is
checked as an anchor.

Ties: the maximum cell is taken in row-major order, earliest first, and
traceback prefers diagonal, then up, then left, so both language arms
report the same alignment.

``sub_matrix``, when given, is indexed by the sorted union of the
symbols in the two sequences; when omitted, matches score +1 and
mismatches -1.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["smith_waterman"]

GAP = "-"


def as_symbols(s):
    if isinstance(s, str):
        return list(s)
    return [str(v) for v in s]


def score_fn(sub_matrix, alpha):
    if sub_matrix is None:
        return lambda x, y: 1.0 if x == y else -1.0
    pos = {}
    for i, c in enumerate(alpha):
        pos[c] = i
    M = [[float(v) for v in r] for r in sub_matrix]
    if len(M) != len(alpha) or any(len(r) != len(alpha) for r in M):
        raise ValueError("smith_waterman: sub_matrix must be square over the symbol alphabet")
    return lambda x, y: M[pos[x]][pos[y]]


def smith_waterman(seq1, seq2, sub_matrix=None, gap=1.0):
    """Optimal local alignment of two sequences.

    Returns
    -------
    score : the optimal local alignment score, never negative
    aligned1, aligned2 : the gapped local segments
    start1, end1, start2, end2 : 1-based inclusive spans in each input
    """
    a = as_symbols(seq1)
    b = as_symbols(seq2)
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        raise ValueError("smith_waterman: neither sequence may be empty")
    g = float(gap)
    if g < 0.0:
        raise ValueError("smith_waterman: gap must be non-negative")
    alpha = sorted(set(a) | set(b))
    s = score_fn(sub_matrix, alpha)
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    bi = 0
    bj = 0
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = H[i - 1][j - 1] + s(a[i - 1], b[j - 1])
            u = H[i - 1][j] - g
            l = H[i][j - 1] - g
            v = 0.0
            if d > v:
                v = d
            if u > v:
                v = u
            if l > v:
                v = l
            H[i][j] = v
            if v > best:
                best = v
                bi = i
                bj = j
    o1 = []
    o2 = []
    i = bi
    j = bj
    while i > 0 and j > 0 and H[i][j] > 0.0:
        if H[i][j] == H[i - 1][j - 1] + s(a[i - 1], b[j - 1]):
            o1.append(a[i - 1])
            o2.append(b[j - 1])
            i -= 1
            j -= 1
        elif H[i][j] == H[i - 1][j] - g:
            o1.append(a[i - 1])
            o2.append(GAP)
            i -= 1
        else:
            o1.append(GAP)
            o2.append(b[j - 1])
            j -= 1
    o1.reverse()
    o2.reverse()
    nm = 0
    nx = 0
    ng = 0
    for p in range(len(o1)):
        if o1[p] == GAP or o2[p] == GAP:
            ng += 1
        elif o1[p] == o2[p]:
            nm += 1
        else:
            nx += 1
    return RichResult(
        title="Smith-Waterman local alignment",
        summary_lines=[("score", best), ("length", len(o1))],
        payload={
            "score": best,
            "estimate": best,
            "aligned1": "".join(o1),
            "aligned2": "".join(o2),
            "length": len(o1),
            "start1": i + 1,
            "end1": bi,
            "start2": j + 1,
            "end2": bj,
            "n_match": nm,
            "n_mismatch": nx,
            "n_gap": ng,
            "n": n,
            "m": m,
            "method": "Smith and Waterman (1981) local DP, zero floor, traceback from the maximum",
        },
    )


def cheatsheet():
    return "alnsw: Smith-Waterman local alignment"


# compact alias per ledger/NAMING.md
smithwaterman = smith_waterman
