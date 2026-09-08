# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Needleman-Wunsch global sequence alignment.

Source: Needleman, S. B. and Wunsch, C. D. (1970), "A general method
applicable to the search for similarities in the amino acid sequence of
two proteins", *Journal of Molecular Biology* 48(3), 443-453,
doi:10.1016/0022-2836(70)90057-4 (citation verified against Crossref).

The dynamic program fills an (n+1) x (m+1) matrix whose entry F(i, j) is
the best score of an alignment of the first i symbols of seq1 against
the first j of seq2:

    F(0, 0) = 0,   F(i, 0) = -i g,   F(0, j) = -j g,
    F(i, j) = max{ F(i-1, j-1) + s(a_i, b_j),
                   F(i-1, j) - g,
                   F(i,   j-1) - g }.

*Global* is the operative word: the first row and column carry the full
gap penalty, so every symbol of both sequences must be placed.  Zeroing
that border is the single edit that turns this into a semi-global
(overlap) alignment, and returns a different, usually larger, number for
the same inputs -- which is why the border initialisation is asserted
directly in the tests rather than only through the final score.

Traceback prefers, on ties, the diagonal, then the up move, then the
left move.  The preference is arbitrary but it is *fixed*, so the two
language arms return the same alignment and not merely the same score.

``sub_matrix``, when given, is indexed by the sorted union of the
symbols occurring in the two sequences; when omitted, matches score +1
and mismatches -1.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["needleman_wunsch"]

GAP = "-"


def as_symbols(s):
    """A sequence as a list of one-character symbols."""
    if isinstance(s, str):
        return list(s)
    return [str(v) for v in s]


def alphabet_of(a, b):
    """Sorted union of the symbols present, which indexes ``sub_matrix``."""
    return sorted(set(a) | set(b))


def score_fn(sub_matrix, alpha):
    """Substitution score s(x, y); default +1 for a match, -1 otherwise."""
    if sub_matrix is None:
        return lambda x, y: 1.0 if x == y else -1.0
    pos = {}
    for i, c in enumerate(alpha):
        pos[c] = i
    M = [[float(v) for v in r] for r in sub_matrix]
    if len(M) != len(alpha) or any(len(r) != len(alpha) for r in M):
        raise ValueError("needleman_wunsch: sub_matrix must be square over the symbol alphabet")
    return lambda x, y: M[pos[x]][pos[y]]


def needleman_wunsch(seq1, seq2, sub_matrix=None, gap=1.0):
    """Optimal global alignment of two sequences.

    Parameters
    ----------
    seq1, seq2 : str or sequence of symbols
        The sequences; neither may be empty.
    sub_matrix : array-like, optional
        Square substitution scores over ``sorted(set(seq1) | set(seq2))``.
    gap : float
        Linear gap penalty, subtracted per gapped position; must be
        non-negative.

    Returns
    -------
    score : the optimal global alignment score
    aligned1, aligned2 : the two gapped strings
    n_match, n_mismatch, n_gap : the composition of the alignment
    """
    a = as_symbols(seq1)
    b = as_symbols(seq2)
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        raise ValueError("needleman_wunsch: neither sequence may be empty")
    g = float(gap)
    if g < 0.0:
        raise ValueError("needleman_wunsch: gap must be non-negative")
    alpha = alphabet_of(a, b)
    s = score_fn(sub_matrix, alpha)
    F = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        F[i][0] = -g * i
    for j in range(1, m + 1):
        F[0][j] = -g * j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = F[i - 1][j - 1] + s(a[i - 1], b[j - 1])
            u = F[i - 1][j] - g
            l = F[i][j - 1] - g
            best = d
            if u > best:
                best = u
            if l > best:
                best = l
            F[i][j] = best
    o1 = []
    o2 = []
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and F[i][j] == F[i - 1][j - 1] + s(a[i - 1], b[j - 1]):
            o1.append(a[i - 1])
            o2.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and F[i][j] == F[i - 1][j] - g:
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
        title="Needleman-Wunsch global alignment",
        summary_lines=[("score", F[n][m]), ("length", len(o1))],
        payload={
            "score": F[n][m],
            "estimate": F[n][m],
            "aligned1": "".join(o1),
            "aligned2": "".join(o2),
            "length": len(o1),
            "n_match": nm,
            "n_mismatch": nx,
            "n_gap": ng,
            "n": n,
            "m": m,
            "method": "Needleman and Wunsch (1970) global DP with linear gap penalty",
        },
    )


def cheatsheet():
    return "alnnw: Needleman-Wunsch global alignment"


# compact alias per ledger/NAMING.md
needlemanwunsch = needleman_wunsch
