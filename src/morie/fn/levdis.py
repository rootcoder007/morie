# morie.fn -- function file (rootcoder007/morie)
"""Levenshtein edit distance."""

from __future__ import annotations

__all__ = ["levenshtein"]


def levenshtein(s1, s2, insert=1.0, delete=1.0, substitute=1.0):
    """Levenshtein distance between two sequences.

    Formula: the dynamic program of Wagner and Fischer,

        ``D[0,0] = 0``,  ``D[i,0] = i c_del``,  ``D[0,j] = j c_ins``,
        ``D[i,j] = min(D[i-1,j] + c_del,
                       D[i,j-1] + c_ins,
                       D[i-1,j-1] + c_sub [s1_i != s2_j])``

    with the answer at ``D[m,n]``.

    Only the previous row is kept, so the memory is ``O(n)`` rather than
    ``O(mn)``; the recurrence is unchanged.  The inputs are compared
    element-by-element, so any sequence of comparable items works, not
    just strings -- which is what makes this usable on token lists.
    With the default unit costs the result is the metric Levenshtein
    defined; unequal costs break symmetry, which is intentional and the
    caller's business.

    Parameters
    ----------
    s1, s2 : sequence
        Sequences to compare.
    insert, delete, substitute : float
        Edit costs.

    Returns
    -------
    int or float
        The edit distance.

    References
    ----------
    Levenshtein (1966), Binary codes capable of correcting deletions,
    insertions and reversals, Soviet Physics Doklady 10:707-710 (the
    metric); Wagner and Fischer (1974), The string-to-string correction
    problem, JACM 21:168-173 (the dynamic program used here).  Neither
    was fetchable -- Doklady is not online and JACM is paywalled -- so
    this is the standard published recurrence, which is anchored in the
    test harness against R's own ``utils::adist``, an independent
    implementation of the same distance.
    """
    a = list(s1)
    b = list(s2)
    m, n = len(a), len(b)
    if m == 0:
        return n * insert
    if n == 0:
        return m * delete
    prev = [j * insert for j in range(n + 1)]
    for i in range(1, m + 1):
        cur = [i * delete] + [0.0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0.0 if ai == b[j - 1] else substitute
            cur[j] = min(prev[j] + delete, cur[j - 1] + insert, prev[j - 1] + cost)
        prev = cur
    d = prev[n]
    return int(d) if float(d).is_integer() else d


def cheatsheet():
    return "levenshtein(s1, s2): Wagner-Fischer edit distance."


# compact alias per ledger/NAMING.md
editdist = levenshtein
