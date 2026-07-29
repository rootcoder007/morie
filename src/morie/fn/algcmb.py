# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Algebraic combinatorics: tableaux, RSK, and counting up to symmetry.

Sagan BE (2001), *The Symmetric Group*, 2nd ed., Springer; Stanley RP
(1999), *Enumerative Combinatorics* vol. 2, Ch 7. Original sources:
Frame JS, Robinson G de B, Thrall RM (1954) *Canadian Journal of
Mathematics* 6:316-324 (hook length); Robinson G de B (1938),
Schensted C (1961) *Canad J Math* 13:179-191, Knuth DE (1970) *Pacific
J Math* 34(3):709-727 (RSK); Burnside W (1897); Polya G (1937).

Two results here are *bijections* rather than formulas, which makes
them testable in a way a count is not: RSK is checked by round-tripping
every permutation of small n and confirming the shapes agree, and its
corollary :math:`\\sum_\\lambda (f^\\lambda)^2 = n!` is checked as an
identity. Burnside's lemma is checked against orbits counted directly.

The Polya here is Polya *enumeration*, counting orbits under a group
action. It is unrelated to the Polya *tree* priors in
``R/ghosal_native.R``, which are Bayesian nonparametrics and named
after the same mathematician for a different reason.
"""

import math
from itertools import permutations, product

from ._richresult import RichResult

__all__ = [
    "hook_lengths",
    "standard_tableaux_count",
    "rsk_insert",
    "rsk_correspondence",
    "rsk_inverse",
    "burnside_orbit_count",
    "cycle_index_necklaces",
    "partitions_of",
]

_METHOD = "Algebraic combinatorics: tableaux, RSK and group actions"


def partitions_of(n, max_part=None):
    """All partitions of ``n``, as weakly decreasing tuples."""
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if n == 0:
        return [()]
    cap = n if max_part is None else min(int(max_part), n)
    out = []
    for first in range(cap, 0, -1):
        for rest in partitions_of(n - first, first):
            out.append((first,) + rest)
    return out


def hook_lengths(shape):
    r"""The hook length of every cell of a Young diagram.

    The hook of a cell is the cell itself, everything to its right in
    the row, and everything below it in the column:

    .. math::
        h(i, j) = \lambda_i - j + \lambda'_j - i + 1.

    Returns
    -------
    dict with ``hooks`` (row-major), ``product``, ``shape``.

    Examples
    --------
    >>> hook_lengths((3, 2))["hooks"]
    [[4, 3, 1], [2, 1]]
    """
    lam = [int(x) for x in shape]
    if any(x <= 0 for x in lam):
        raise ValueError(f"every part must be positive; got {shape}.")
    if any(a < b for a, b in zip(lam, lam[1:])):
        raise ValueError(
            f"a partition must be weakly decreasing; got {shape}."
        )
    conj = [sum(1 for r in lam if r > j) for j in range(lam[0] if lam else 0)]
    hooks = []
    for i, row in enumerate(lam):
        hooks.append([row - j + conj[j] - i - 1 for j in range(row)])
    prod = 1
    for r in hooks:
        for h in r:
            prod *= h
    return {"hooks": hooks, "product": prod, "shape": tuple(lam),
            "conjugate": tuple(conj), "n": sum(lam)}


def standard_tableaux_count(shape):
    r"""The hook length formula:
    :math:`f^\lambda = n! / \prod_{(i,j)} h(i,j)`.

    Counts standard Young tableaux of the given shape -- fillings by
    :math:`1, \ldots, n` increasing along every row and down every
    column. That a product of hook lengths should divide :math:`n!`
    exactly is not obvious, and the payload reports the remainder so
    the division is checked rather than assumed.

    Examples
    --------
    >>> standard_tableaux_count((3, 2))["count"]
    5
    >>> standard_tableaux_count((2, 2))["count"]
    2
    >>> standard_tableaux_count((4,))["count"]
    1
    """
    h = hook_lengths(shape)
    n = h["n"]
    fact = math.factorial(n)
    prod = h["product"]
    q, r = divmod(fact, prod)
    out = RichResult(
        title=f"Standard Young tableaux of shape {h['shape']}",
        summary_lines=[
            ("Count", q),
            ("n!", fact),
            ("Product of hooks", prod),
            ("Divides exactly", r == 0),
        ],
        payload={
            "count": q,
            "estimate": float(q),
            "exact": str(q),
            "factorial": fact,
            "hook_product": prod,
            "remainder": r,
            "divides_exactly": r == 0,
            "hooks": h["hooks"],
            "shape": h["shape"],
            "n": n,
            "method": "Hook length formula (Frame, Robinson and Thrall 1954)",
        },
    )
    if r != 0:
        out.warnings.append(
            f"The hook product {prod} does not divide {n}! exactly "
            f"(remainder {r}), which cannot happen for a genuine partition. "
            "The shape or the hook computation is wrong."
        )
    return out


def rsk_insert(tableau, value):
    """Row-insert one value into a semistandard tableau (bumping).

    Returns the new tableau and the row index where a cell was added.
    """
    T = [list(r) for r in tableau]
    x = value
    for i, row in enumerate(T):
        # find the leftmost entry strictly greater than x
        pos = None
        for j, v in enumerate(row):
            if v > x:
                pos = j
                break
        if pos is None:
            row.append(x)
            return T, i
        row[pos], x = x, row[pos]
    T.append([x])
    return T, len(T) - 1


def rsk_correspondence(permutation):
    r"""The Robinson-Schensted correspondence.

    Maps a permutation to a pair of standard Young tableaux of the
    *same* shape, bijectively. The shape is what makes it useful: the
    first row length is the longest increasing subsequence and the
    first column length is the longest decreasing one, so a purely
    combinatorial statistic falls out of an algebraic construction.

    Returns
    -------
    RichResult with ``p_tableau``, ``q_tableau``, ``shape``,
    ``longest_increasing``, ``longest_decreasing``, ``same_shape``.

    Examples
    --------
    >>> out = rsk_correspondence([3, 1, 2])
    >>> out["shape"]
    (2, 1)
    >>> out["longest_increasing"]
    2
    """
    w = [int(x) for x in permutation]
    n = len(w)
    if sorted(w) != list(range(1, n + 1)):
        raise ValueError(
            f"expected a permutation of 1..{n}; got {permutation}."
        )
    P, Q = [], []
    for step, x in enumerate(w, start=1):
        P, row = rsk_insert(P, x)
        while len(Q) <= row:
            Q.append([])
        Q[row].append(step)
    shape = tuple(len(r) for r in P)
    qshape = tuple(len(r) for r in Q)
    inc = shape[0] if shape else 0
    dec = len(shape)
    return RichResult(
        title="Robinson-Schensted correspondence",
        summary_lines=[
            ("Shape", shape),
            ("Longest increasing subsequence", inc),
            ("Longest decreasing subsequence", dec),
            ("P and Q share a shape", shape == qshape),
        ],
        payload={
            "p_tableau": P,
            "q_tableau": Q,
            "shape": shape,
            "q_shape": qshape,
            "same_shape": shape == qshape,
            "longest_increasing": inc,
            "longest_decreasing": dec,
            "estimate": float(inc),
            "permutation": w,
            "n": n,
            "method": "Robinson-Schensted (Robinson 1938; Schensted 1961)",
        },
    )


def rsk_inverse(p_tableau, q_tableau):
    """Recover the permutation from its two tableaux.

    RSK is a bijection, so this must return the original word exactly.
    Round-tripping is the strongest available check on the forward
    map, and is what the tests use.
    """
    P = [list(r) for r in p_tableau]
    Q = [list(r) for r in q_tableau]
    if [len(r) for r in P] != [len(r) for r in Q]:
        raise ValueError("P and Q must have the same shape.")
    n = sum(len(r) for r in P)
    word = []
    for step in range(n, 0, -1):
        # locate `step` in Q; it is at the end of some row
        row = None
        for i, r in enumerate(Q):
            if r and r[-1] == step:
                row = i
                break
        if row is None:
            raise ValueError(f"{step} is not at the end of any row of Q.")
        Q[row].pop()
        x = P[row].pop()
        for i in range(row - 1, -1, -1):
            # reverse-bump: the rightmost entry strictly less than x
            pos = None
            for j in range(len(P[i]) - 1, -1, -1):
                if P[i][j] < x:
                    pos = j
                    break
            P[i][pos], x = x, P[i][pos]
        word.append(x)
        while P and not P[-1]:
            P.pop()
        while Q and not Q[-1]:
            Q.pop()
    return list(reversed(word))


def burnside_orbit_count(group_permutations, n_colours):
    r"""Burnside's lemma: the number of orbits is the average number of
    fixed points.

    .. math::
        |X/G| = \frac{1}{|G|}\sum_{g \in G} |X^g|,

    and for colourings of positions, :math:`|X^g| = k^{c(g)}` where
    :math:`c(g)` is the number of cycles of :math:`g`.

    The lemma is often stated as "divide by the symmetry", which is
    wrong whenever some arrangements have extra symmetry of their own.
    ``naive_division`` is returned to make the gap visible: for
    two-colour necklaces of length 4 there are 16 colourings and 4
    rotations, but 6 orbits rather than 4.

    Returns
    -------
    RichResult with ``orbits``, ``fixed_points``, ``naive_division``,
    ``naive_is_wrong``.

    Examples
    --------
    >>> rot4 = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    >>> out = burnside_orbit_count(rot4, 2)
    >>> out["orbits"], out["naive_division"]
    (6, 4.0)
    """
    G = [list(map(int, g)) for g in group_permutations]
    k = int(n_colours)
    if not G:
        raise ValueError("the group must contain at least the identity.")
    n = len(G[0])
    if any(len(g) != n for g in G):
        raise ValueError("every group element must permute the same set.")
    if k < 1:
        raise ValueError(f"n_colours must be positive; got {k}.")
    for g in G:
        if sorted(g) != list(range(n)):
            raise ValueError(
                f"{g} is not a permutation of 0 .. {n - 1}."
            )

    def n_cycles(g):
        seen, c = set(), 0
        for i in range(n):
            if i in seen:
                continue
            c += 1
            j = i
            while j not in seen:
                seen.add(j)
                j = g[j]
        return c

    cyc = [n_cycles(g) for g in G]
    fixed = [k ** c for c in cyc]
    total = sum(fixed)
    q, r = divmod(total, len(G))
    naive = k ** n / len(G)
    out = RichResult(
        title="Burnside orbit count",
        summary_lines=[
            ("Orbits", q),
            ("Total colourings", k ** n),
            ("Group order", len(G)),
            ("Naive |X|/|G|", naive),
        ],
        payload={
            "orbits": q,
            "estimate": float(q),
            "exact": str(q),
            "fixed_points": fixed,
            "cycle_counts": cyc,
            "sum_fixed": total,
            "group_order": len(G),
            "total_colourings": k ** n,
            "naive_division": naive,
            "naive_is_wrong": abs(naive - q) > 1e-12,
            "divides_exactly": r == 0,
            "n": n,
            "method": "Burnside's lemma",
        },
    )
    if r != 0:
        out.warnings.append(
            f"The fixed-point total {total} is not divisible by the group "
            f"order {len(G)}, which is impossible if the supplied "
            "permutations really form a group. Check closure."
        )
    return out


def cycle_index_necklaces(n, k):
    r"""Necklaces: colourings of :math:`n` beads with :math:`k` colours
    up to rotation.

    By Burnside applied to the cyclic group,

    .. math:: \frac{1}{n}\sum_{d \mid n} \varphi(n/d)\, k^{d}.

    Computed both that way and by direct Burnside over the explicit
    rotations, and the two are compared -- the closed form is a
    shortcut, and shortcuts are where errors hide.

    Examples
    --------
    >>> [cycle_index_necklaces(n, 2)["count"] for n in range(1, 7)]
    [2, 3, 4, 6, 8, 14]
    """
    n, k = int(n), int(k)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    if k < 1:
        raise ValueError(f"k must be positive; got {k}.")

    def phi(m):
        r = m
        p = 2
        mm = m
        while p * p <= mm:
            if mm % p == 0:
                while mm % p == 0:
                    mm //= p
                r -= r // p
            p += 1
        if mm > 1:
            r -= r // mm
        return r

    total = sum(phi(n // d) * k ** d for d in range(1, n + 1) if n % d == 0)
    q, rem = divmod(total, n)
    rot = [[(i + s) % n for i in range(n)] for s in range(n)]
    direct = burnside_orbit_count(rot, k)["orbits"]
    return RichResult(
        title=f"Necklaces, {n} beads, {k} colours",
        summary_lines=[
            ("Count", q),
            ("Direct Burnside", direct),
            ("Agree", q == direct),
        ],
        payload={
            "count": q,
            "estimate": float(q),
            "exact": str(q),
            "direct_burnside": direct,
            "agrees": q == direct,
            "divides_exactly": rem == 0,
            "total_colourings": k ** n,
            "n": n,
            "k": k,
            "method": "Cycle index of the cyclic group (Polya enumeration)",
        },
    )


def cheatsheet():
    return (
        "algcmb: hook length formula, the Robinson-Schensted bijection with "
        "its inverse, Burnside's lemma and necklace counting -- bijections "
        "checked by round-tripping, counts by direct enumeration"
    )
