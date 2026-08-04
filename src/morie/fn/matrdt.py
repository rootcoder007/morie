# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Matroids: the structure that makes the greedy algorithm correct.

Oxley JG (2011), *Matroid Theory*, 2nd ed., Oxford University Press;
Welsh DJA (1976), *Matroid Theory*, Academic Press. Original sources:
Whitney H (1935) *American Journal of Mathematics* 57(3):509-533;
Rado R (1957); Edmonds J (1971) *Mathematical Programming* 1:127-136.

The organising result is Rado and Edmonds': **the greedy algorithm
returns a maximum-weight independent set for every weighting if and
only if the independence system is a matroid.** That is an equivalence,
so it can be tested in both directions -- greedy is confirmed optimal
on matroids and confirmed to *fail* on a hand-built independence system
that satisfies heredity but not exchange. A test that only checks the
forward direction would pass on code that had no idea what a matroid
was.
"""

import math
from itertools import chain, combinations

from ._richresult import RichResult

__all__ = [
    "is_matroid",
    "matroid_rank",
    "matroid_bases",
    "matroid_circuits",
    "matroid_dual",
    "greedy_independent_set",
    "uniform_matroid",
    "graphic_matroid",
    "brute_force_max_weight",
]

_METHOD = "Matroid structure and greedy optimality"


def _subsets(ground):
    g = list(ground)
    return chain.from_iterable(combinations(g, r) for r in range(len(g) + 1))


def _norm(family):
    return {frozenset(s) for s in family}


def is_matroid(ground, independent):
    r"""Check the two matroid axioms directly.

    An independence system on a finite ground set is a matroid when

    1. **heredity** -- every subset of an independent set is
       independent, and
    2. **exchange** -- if :math:`|A| < |B|` with both independent, some
       element of :math:`B \setminus A` can be added to :math:`A`
       keeping it independent.

    Both are checked exhaustively, and the first violating pair is
    returned rather than a bare False, because "not a matroid" without
    a witness is not a useful answer.

    Returns
    -------
    RichResult with ``is_matroid``, ``hereditary``, ``exchange``,
    ``heredity_violation``, ``exchange_violation``.

    Examples
    --------
    >>> g = [0, 1, 2]
    >>> ind = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    >>> is_matroid(g, ind)["is_matroid"]          # uniform U(2,3)
    True
    """
    g = list(ground)
    fam = _norm(independent)
    if frozenset() not in fam:
        return RichResult(
            title="Matroid axiom check",
            payload={"is_matroid": False, "hereditary": False,
                     "exchange": None, "estimate": 0.0,
                     "heredity_violation": "the empty set is not independent",
                     "exchange_violation": None, "n": len(g),
                     "method": _METHOD},
            summary_lines=[("Is a matroid", False)],
        )

    hered_bad = None
    for s in fam:
        for e in s:
            if (s - {e}) not in fam:
                hered_bad = (sorted(s), sorted(s - {e}))
                break
        if hered_bad:
            break

    exch_bad = None
    if hered_bad is None:
        for a in fam:
            for b in fam:
                if len(a) >= len(b):
                    continue
                if not any((a | {x}) in fam for x in (b - a)):
                    exch_bad = (sorted(a), sorted(b))
                    break
            if exch_bad:
                break

    ok = hered_bad is None and exch_bad is None
    out = RichResult(
        title="Matroid axiom check",
        summary_lines=[
            ("Is a matroid", ok),
            ("Hereditary", hered_bad is None),
            ("Exchange", exch_bad is None if hered_bad is None else None),
            ("Independent sets", len(fam)),
        ],
        payload={
            "is_matroid": ok,
            "estimate": float(ok),
            "hereditary": hered_bad is None,
            "exchange": (exch_bad is None) if hered_bad is None else None,
            "heredity_violation": hered_bad,
            "exchange_violation": exch_bad,
            "n_independent": len(fam),
            "n": len(g),
            "method": _METHOD,
        },
    )
    if hered_bad:
        out.warnings.append(
            f"Heredity fails: {hered_bad[0]} is independent but its subset "
            f"{hered_bad[1]} is not. This is not even an independence system."
        )
    if exch_bad:
        out.warnings.append(
            f"Exchange fails: {exch_bad[0]} and {exch_bad[1]} are both "
            "independent with the first smaller, yet no element of the second "
            "can be added to the first. The greedy algorithm is therefore not "
            "guaranteed optimal on this system."
        )
    return out


def matroid_rank(ground, independent, subset=None):
    """Rank of a subset: the size of its largest independent subset.

    Returns the rank of the whole ground set when ``subset`` is None.
    """
    fam = _norm(independent)
    target = frozenset(ground if subset is None else subset)
    best = 0
    for s in fam:
        if s <= target:
            best = max(best, len(s))
    return best


def matroid_bases(ground, independent):
    """The maximal independent sets. All have the same size in a
    matroid -- which is a consequence of exchange, not an assumption."""
    fam = _norm(independent)
    r = matroid_rank(ground, independent)
    return sorted((sorted(s) for s in fam if len(s) == r), key=tuple)


def matroid_circuits(ground, independent):
    """The minimal dependent sets."""
    g = list(ground)
    fam = _norm(independent)
    dep = [frozenset(s) for s in _subsets(g) if frozenset(s) not in fam]
    circuits = []
    for d in dep:
        if all(not (frozenset(c) < d) for c in dep if frozenset(c) != d):
            circuits.append(sorted(d))
    return sorted(circuits, key=lambda c: (len(c), c))


def matroid_dual(ground, independent):
    r"""The dual matroid: independent sets are the complements of
    spanning sets.

    :math:`X` is independent in :math:`M^*` exactly when
    :math:`E \setminus X` contains a basis of :math:`M`. Duality is an
    involution, which the tests check by dualising twice.
    """
    g = list(ground)
    E = frozenset(g)
    bases = [frozenset(b) for b in matroid_bases(g, independent)]
    dual = []
    for s in _subsets(g):
        rest = E - frozenset(s)
        if any(b <= rest for b in bases):
            dual.append(sorted(s))
    return sorted(dual, key=lambda x: (len(x), x))


def uniform_matroid(n, k):
    r"""The uniform matroid :math:`U_{k,n}`: every subset of size at
    most :math:`k` is independent.

    Examples
    --------
    >>> u = uniform_matroid(3, 2)
    >>> len(u["independent"])
    7
    """
    n, k = int(n), int(k)
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative.")
    g = list(range(n))
    ind = [list(s) for s in _subsets(g) if len(s) <= k]
    return {"ground": g, "independent": ind, "rank": min(k, n),
            "name": f"U({k},{n})"}


def graphic_matroid(edges, n_vertices):
    r"""The cycle matroid of a graph: a set of edges is independent
    exactly when it contains no cycle -- that is, when it is a forest.

    This is the matroid that makes Kruskal's algorithm correct, and the
    connection is not decorative: minimum spanning tree *is* the greedy
    algorithm on this matroid.
    """
    E = [tuple(sorted(e)) for e in edges]
    m = len(E)
    n = int(n_vertices)

    def acyclic(sub):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in sub:
            a, b = E[i]
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
        return True

    ind = [list(s) for s in _subsets(range(m)) if acyclic(s)]
    return {"ground": list(range(m)), "independent": ind, "edges": E,
            "n_vertices": n, "name": "graphic"}


def greedy_independent_set(ground, independent, weights):
    r"""The greedy algorithm: sort by weight, take what keeps
    independence.

    Only elements of positive weight are taken, since adding a
    negative-weight element can never improve a maximum-weight
    independent set.
    """
    fam = _norm(independent)
    g = list(ground)
    w = dict(zip(g, weights)) if not isinstance(weights, dict) else weights
    order = sorted(g, key=lambda e: (-w.get(e, 0), e))
    cur = frozenset()
    for e in order:
        if w.get(e, 0) <= 0:
            continue
        if (cur | {e}) in fam:
            cur = cur | {e}
    return {"set": sorted(cur), "weight": sum(w.get(e, 0) for e in cur)}


def brute_force_max_weight(ground, independent, weights):
    """Maximum-weight independent set by exhaustive search."""
    fam = _norm(independent)
    g = list(ground)
    w = dict(zip(g, weights)) if not isinstance(weights, dict) else weights
    best, best_set = None, None
    for s in fam:
        tot = sum(w.get(e, 0) for e in s)
        if best is None or tot > best:
            best, best_set = tot, sorted(s)
    return {"set": best_set, "weight": best if best is not None else 0}


def cheatsheet():
    return (
        "matrdt: matroid axioms with a witness on failure, rank, bases, "
        "circuits, duality, and the Rado-Edmonds greedy theorem tested in "
        "both directions"
    )


# compact alias per ledger/NAMING.md
graphicmatroid = graphic_matroid


# compact alias per ledger/NAMING.md
ismatroid = is_matroid


# compact alias per ledger/NAMING.md
matroidbases = matroid_bases


# compact alias per ledger/NAMING.md
matroiddual = matroid_dual


# compact alias per ledger/NAMING.md
matroidrank = matroid_rank


# compact alias per ledger/NAMING.md
uniformmatroid = uniform_matroid
