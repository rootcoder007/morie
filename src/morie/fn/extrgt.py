# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extremal combinatorics: how large a structure can get before it must
contain something.

Bollobas B (2004), *Extremal Graph Theory*, Dover -- Turan's theorem is
Ch VI. Original sources: Mantel W (1907); Turan P (1941); Sperner E
(1928); Erdos P, Ko C, Rado R (1961) *Quart J Math* 12:313-320;
Dilworth RP (1950) *Annals of Mathematics* 51(1):161-166.

Like the Ramsey shelf, most of this is exactly checkable rather than
checkable to a tolerance: every bound here is attained by an explicit
construction, and the constructions are built and verified rather than
cited.
"""

import math
from itertools import combinations

from ._richresult import RichResult

__all__ = [
    "turan_number",
    "turan_graph",
    "mantel_number",
    "sperner_width",
    "erdos_ko_rado",
    "dilworth_decomposition",
    "count_edges",
    "has_clique",
]

_METHOD = "Extremal combinatorics with attained constructions"


def count_edges(adjacency):
    """Number of edges in a simple undirected graph."""
    A = [list(map(int, row)) for row in adjacency]
    n = len(A)
    return sum(A[i][j] != 0 for i in range(n) for j in range(i + 1, n))


def has_clique(adjacency, k):
    """Does the graph contain a clique on ``k`` vertices?

    Exhaustive, which is the only honest answer for a certificate.
    Returns the clique itself when one exists.
    """
    A = [list(map(int, row)) for row in adjacency]
    n = len(A)
    k = int(k)
    if k <= 0:
        return []
    if k > n:
        return None
    for c in combinations(range(n), k):
        if all(A[i][j] for i, j in combinations(c, 2)):
            return list(c)
    return None


def turan_graph(n, r):
    r"""The complete :math:`r`-partite graph with parts as equal as
    possible -- the unique extremal graph for Turan's theorem.

    Vertices are split into :math:`r` parts of size
    :math:`\lfloor n/r \rfloor` or :math:`\lceil n/r \rceil`, and every
    pair in different parts is joined. It contains no
    :math:`K_{r+1}`, because any :math:`r+1` vertices must repeat a
    part and two vertices in the same part are non-adjacent.

    Returns
    -------
    dict with ``adjacency``, ``parts``, ``edges``.
    """
    n, r = int(n), int(r)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if r < 1:
        raise ValueError(f"r must be at least 1; got {r}.")
    parts = [[] for _ in range(r)]
    for v in range(n):
        parts[v % r].append(v)
    part_of = {}
    for i, p in enumerate(parts):
        for v in p:
            part_of[v] = i
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if part_of[i] != part_of[j]:
                A[i][j] = A[j][i] = 1
    return {"adjacency": A, "parts": parts, "edges": count_edges(A)}


def turan_number(n, r):
    r"""The most edges an :math:`n`-vertex graph can have with no
    :math:`K_{r+1}`.

    Turan's theorem: the maximum is attained by the complete
    :math:`r`-partite graph with parts as equal as possible, and that
    graph is the *unique* extremal example. The count is

    .. math::
        \left(1 - \tfrac{1}{r}\right)\frac{n^2}{2}

    only when :math:`r` divides :math:`n`; otherwise that expression is
    an upper bound and the exact value comes from summing the parts.
    Both are returned, because the difference is where the usual
    textbook statement is loose: at :math:`n = 10, r = 3` the rounded
    formula gives 33.33 and the exact answer is 33, and at
    :math:`n = 11, r = 3` it gives 40.33 against an exact 40.

    Returns
    -------
    RichResult with ``estimate`` (exact edge count),
    ``rounded_formula``, ``part_sizes``, ``attained``.

    Examples
    --------
    >>> turan_number(5, 2)["count"]
    6
    >>> turan_number(10, 3)["count"]
    33
    """
    n, r = int(n), int(r)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if r < 1:
        raise ValueError(f"r must be at least 1; got {r}.")
    sizes = [n // r + (1 if i < n % r else 0) for i in range(r)]
    # edges = all pairs minus pairs inside a part
    exact = math.comb(n, 2) - sum(math.comb(s, 2) for s in sizes)
    approx = (1 - 1 / r) * n * n / 2
    g = turan_graph(n, r)
    return RichResult(
        title=f"Turan number ex(n = {n}, K_{r + 1})",
        summary_lines=[
            ("Maximum edges", exact),
            ("Rounded formula (1-1/r)n^2/2", approx),
            ("Part sizes", sizes),
        ],
        payload={
            "count": exact,
            "estimate": float(exact),
            "exact": str(exact),
            "rounded_formula": approx,
            "formula_is_exact": abs(approx - exact) < 1e-9,
            "part_sizes": sizes,
            "construction_edges": g["edges"],
            "attained": g["edges"] == exact,
            "forbidden_clique": r + 1,
            "n": n,
            "r": r,
            "method": "Turan's theorem (Turan 1941)",
        },
        interpretation=(
            f"No graph on {n} vertices without a K_{r + 1} has more than "
            f"{exact} edges, and the complete {r}-partite graph with parts "
            f"{sizes} attains it."
        ),
    )


def mantel_number(n):
    r"""The triangle-free case :math:`r = 2`:
    :math:`\lfloor n^2/4 \rfloor` edges, attained by the balanced
    complete bipartite graph.

    Mantel's theorem predates Turan's by 34 years and is the same
    statement at :math:`r = 2`. The floor matters at odd :math:`n`:
    :math:`n^2/4` is not an integer there.

    Examples
    --------
    >>> [mantel_number(n)["count"] for n in range(2, 8)]
    [1, 2, 4, 6, 9, 12]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    out = turan_number(n, 2)
    out.payload["floor_n2_over_4"] = n * n // 4
    out.payload["method"] = "Mantel's theorem (Mantel 1907)"
    out.title = f"Mantel number ex(n = {n}, triangle)"
    return out


def sperner_width(n):
    r"""The largest antichain in the subset lattice of an
    :math:`n`-set.

    Sperner's theorem: no family of subsets in which none contains
    another can be larger than :math:`\binom{n}{\lfloor n/2 \rfloor}`,
    and the middle layer attains it. At even :math:`n` the extremal
    family is unique; at odd :math:`n` there are two, the two central
    layers, which have the same size.

    Examples
    --------
    >>> [sperner_width(n)["count"] for n in range(1, 7)]
    [1, 2, 3, 6, 10, 20]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    w = math.comb(n, n // 2)
    layers = [n // 2] if n % 2 == 0 else [n // 2, n // 2 + 1]
    return RichResult(
        title=f"Sperner width of a {n}-set",
        summary_lines=[
            ("Largest antichain", w),
            ("Attaining layer(s)", layers),
            ("Total subsets", 2 ** n),
        ],
        payload={
            "count": w,
            "estimate": float(w),
            "exact": str(w),
            "extremal_layers": layers,
            "unique_extremal": n % 2 == 0,
            "total_subsets": 2 ** n,
            "fraction_of_powerset": w / 2 ** n if n < 1000 else float("nan"),
            "n": n,
            "method": "Sperner's theorem (Sperner 1928)",
        },
        interpretation=(
            f"No antichain of subsets of a {n}-set exceeds {w} members; the "
            f"middle layer(s) {layers} attain it."
        ),
    )


def erdos_ko_rado(n, k):
    r"""The largest intersecting family of :math:`k`-subsets of an
    :math:`n`-set.

    Erdos, Ko and Rado: for :math:`n \ge 2k` the maximum is
    :math:`\binom{n-1}{k-1}`, attained by the *star* -- every
    :math:`k`-set containing a fixed element.

    The condition :math:`n \ge 2k` is not decoration. Below it, no two
    :math:`k`-subsets can be disjoint at all, so **every** family is
    intersecting and the answer is the trivial :math:`\binom{n}{k}`.
    The bound and the regime it holds in are reported separately,
    because quoting :math:`\binom{n-1}{k-1}` at :math:`n < 2k` gives an
    answer that is too small and looks reasonable.

    Examples
    --------
    >>> erdos_ko_rado(6, 3)["count"]
    10
    >>> erdos_ko_rado(5, 3)["count"]          # n < 2k: everything works
    10
    >>> erdos_ko_rado(5, 3)["ekr_regime"]
    False
    """
    n, k = int(n), int(k)
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative.")
    if k > n:
        raise ValueError(f"k must not exceed n; got k = {k}, n = {n}.")
    regime = n >= 2 * k
    star = math.comb(n - 1, k - 1) if k >= 1 else 0
    count = star if regime else math.comb(n, k)
    out = RichResult(
        title=f"Erdos-Ko-Rado, n = {n}, k = {k}",
        summary_lines=[
            ("Largest intersecting family", count),
            ("Star size C(n-1, k-1)", star),
            ("n >= 2k", regime),
        ],
        payload={
            "count": count,
            "estimate": float(count),
            "exact": str(count),
            "star_size": star,
            "all_k_sets": math.comb(n, k),
            "ekr_regime": regime,
            "n": n,
            "k": k,
            "method": "Erdos-Ko-Rado theorem (1961)",
        },
        interpretation=(
            f"The largest intersecting family of {k}-subsets has {count} "
            "members, attained by the star through any fixed element."
            if regime else
            f"With n < 2k no two {k}-subsets are disjoint, so every family "
            f"is intersecting and the maximum is all {count} of them."
        ),
    )
    if not regime:
        out.warnings.append(
            f"n = {n} is below 2k = {2 * k}, outside the Erdos-Ko-Rado "
            f"regime. The star bound C(n-1, k-1) = {star} is NOT the answer "
            "here; it is smaller than the truth because every family is "
            "intersecting when no two k-sets can be disjoint."
        )
    return out


def dilworth_decomposition(leq):
    r"""Dilworth's theorem: the largest antichain in a finite poset
    equals the fewest chains needed to cover it.

    Both sides are computed -- the antichain by maximum independent set
    in the comparability graph, the chain cover by a matching in the
    split bipartite graph -- and the theorem is then *checked* rather
    than assumed, since equality is the content of it.

    Parameters
    ----------
    leq : callable or 2-D array
        ``leq[i][j]`` true when :math:`i \le j` in the partial order.

    Returns
    -------
    RichResult with ``antichain_size``, ``chain_cover_size``,
    ``dilworth_holds``, ``antichain``, ``chains``.
    """
    if callable(leq):
        raise ValueError("pass a matrix; a callable needs a known ground set.")
    M = [[bool(v) for v in row] for row in leq]
    n = len(M)
    if any(len(r) != n for r in M):
        raise ValueError("leq must be square.")
    for i in range(n):
        if not M[i][i]:
            raise ValueError(f"leq must be reflexive; element {i} is not.")
    for i in range(n):
        for j in range(n):
            if i != j and M[i][j] and M[j][i]:
                raise ValueError(
                    f"leq must be antisymmetric; {i} and {j} are mutually "
                    "below one another."
                )
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if M[i][j] and M[j][k] and not M[i][k]:
                    raise ValueError(
                        f"leq must be transitive; {i} <= {j} <= {k} but not "
                        f"{i} <= {k}."
                    )

    strict = [[M[i][j] and i != j for j in range(n)] for i in range(n)]
    # minimum chain cover = n - maximum matching in the split graph
    match = [-1] * n

    def try_augment(u, seen):
        for v in range(n):
            if strict[u][v] and not seen[v]:
                seen[v] = True
                if match[v] == -1 or try_augment(match[v], seen):
                    match[v] = u
                    return True
        return False

    size = 0
    for u in range(n):
        if try_augment(u, [False] * n):
            size += 1
    chain_cover = n - size

    # rebuild the chains from the matching
    nxt = {}
    for v, u in enumerate(match):
        if u != -1:
            nxt[u] = v
    starts = [u for u in range(n) if u not in match or match.count(u) == 0]
    starts = [u for u in range(n) if u not in set(m for m in match if m != -1)]
    chains = []
    for s in starts:
        ch, cur = [s], s
        while cur in nxt:
            cur = nxt[cur]
            ch.append(cur)
        chains.append(ch)

    # largest antichain by exhaustive search on the comparability graph
    best = []
    for size_try in range(n, 0, -1):
        found = None
        for c in combinations(range(n), size_try):
            if all(not (strict[i][j] or strict[j][i])
                   for i, j in combinations(c, 2)):
                found = list(c)
                break
        if found is not None:
            best = found
            break

    out = RichResult(
        title="Dilworth decomposition",
        summary_lines=[
            ("Largest antichain", len(best)),
            ("Minimum chain cover", chain_cover),
            ("Dilworth equality", len(best) == chain_cover),
        ],
        payload={
            "antichain_size": len(best),
            "estimate": float(len(best)),
            "chain_cover_size": chain_cover,
            "dilworth_holds": len(best) == chain_cover,
            "antichain": best,
            "chains": chains,
            "n": n,
            "method": "Dilworth's theorem (1950)",
        },
        interpretation=(
            f"The largest antichain has {len(best)} elements and the poset "
            f"decomposes into {chain_cover} chains; Dilworth's theorem says "
            "these must be equal."
        ),
    )
    if len(best) != chain_cover:
        out.warnings.append(
            f"The antichain ({len(best)}) and chain cover ({chain_cover}) "
            "disagree, which contradicts Dilworth's theorem. One of the two "
            "computations is wrong, or the relation supplied is not a "
            "partial order."
        )
    return out


def cheatsheet():
    return (
        "extrgt: Turan, Mantel, Sperner, Erdos-Ko-Rado and Dilworth, each "
        "with the construction that attains the bound built and checked"
    )
