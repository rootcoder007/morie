# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Design theory and coding bounds: what can exist, and what cannot.

Stinson DR (2004), *Combinatorial Designs: Constructions and Analysis*,
Springer; Colbourn CJ, Dinitz JH (2007), *Handbook of Combinatorial
Designs*, 2nd ed., CRC. Original sources: Fisher RA (1940) *Annals of
Eugenics* 10:52-75; Kirkman TP (1847); Bose RC (1939); Hamming RW
(1950) *Bell System Technical Journal* 29:147-160; Singleton RC (1964)
*IEEE Trans Inform Theory* 10(2):116-118.

The useful thing about this material is that most of it is a decision
rather than an estimate. A design with given parameters either can
exist or provably cannot, and the necessary conditions are integer
arithmetic. Where a condition is necessary but not sufficient -- which
is the usual case -- that is stated, because "the conditions hold" is
not the same claim as "the design exists".
"""

import math
from itertools import combinations, permutations

from ._richresult import RichResult

__all__ = [
    "bibd_parameters",
    "steiner_triple_system",
    "latin_square",
    "is_latin_square",
    "are_orthogonal",
    "hamming_bound",
    "singleton_bound",
    "incidence_matrix_check",
]

_METHOD = "Combinatorial design feasibility and coding bounds"


def bibd_parameters(v, k, lam):
    r"""Necessary conditions for a balanced incomplete block design.

    A :math:`(v, k, \lambda)`-BIBD arranges :math:`v` points into
    blocks of size :math:`k` so that every pair of points appears in
    exactly :math:`\lambda` blocks. Counting the same incidences two
    ways forces

    .. math::
        r = \frac{\lambda(v-1)}{k-1}, \qquad b = \frac{vr}{k},

    and both must be whole numbers. Fisher's inequality adds
    :math:`b \ge v`.

    **These are necessary, not sufficient.** The standard
    counterexample is :math:`(22, 7, 2)`: the divisibility conditions
    hold and Fisher's inequality holds, yet no such design exists --
    ruled out by the Bruck-Ryser-Chowla theorem. The payload separates
    ``divisibility_ok``, ``fisher_ok`` and ``exists``, and ``exists``
    is left as None wherever the arithmetic cannot settle it, rather
    than reporting feasibility as existence.

    Returns
    -------
    RichResult with ``r``, ``b``, ``divisibility_ok``, ``fisher_ok``,
    ``feasible``, ``exists``.

    Examples
    --------
    >>> out = bibd_parameters(7, 3, 1)     # the Fano plane
    >>> out["r"], out["b"], out["feasible"]
    (3, 7, True)
    >>> bibd_parameters(8, 3, 1)["feasible"]
    False
    """
    v, k, lam = int(v), int(k), int(lam)
    if v < 1 or k < 1 or lam < 1:
        raise ValueError("v, k and lambda must be positive.")
    if k > v:
        raise ValueError(f"block size k = {k} cannot exceed v = {v}.")
    if k < 2:
        raise ValueError(f"block size k must be at least 2; got {k}.")

    num_r = lam * (v - 1)
    den_r = k - 1
    r_ok = num_r % den_r == 0
    r = num_r // den_r if r_ok else None
    b_ok = False
    b = None
    if r is not None:
        b_ok = (v * r) % k == 0
        b = (v * r) // k if b_ok else None
    div_ok = r_ok and b_ok
    fisher_ok = (b is not None and b >= v)
    feasible = div_ok and fisher_ok

    # a few cases the arithmetic alone cannot settle
    exists = None
    note = ""
    if not feasible:
        exists = False
        note = "ruled out by the counting or Fisher conditions"
    elif (v, k, lam) == (22, 7, 2):
        exists = False
        note = ("ruled out by Bruck-Ryser-Chowla despite passing every "
                "counting condition")
    elif k == v:
        exists = True
        note = "trivial: the single block containing every point"

    out = RichResult(
        title=f"BIBD({v}, {k}, {lam}) feasibility",
        summary_lines=[
            ("r (blocks per point)", r),
            ("b (number of blocks)", b),
            ("Divisibility", div_ok),
            ("Fisher b >= v", fisher_ok),
            ("Feasible", feasible),
            ("Exists", exists),
        ],
        payload={
            "v": v, "k": k, "lambda": lam, "r": r, "b": b,
            "estimate": float(b) if b is not None else float("nan"),
            "divisibility_ok": div_ok,
            "r_integral": r_ok,
            "b_integral": b_ok,
            "fisher_ok": fisher_ok,
            "feasible": feasible,
            "exists": exists,
            "note": note,
            "n": v,
            "method": "BIBD necessary conditions with Fisher's inequality",
        },
        interpretation=(
            f"A ({v}, {k}, {lam})-BIBD would need r = {r} and b = {b}."
            if feasible else
            f"No ({v}, {k}, {lam})-BIBD can exist: the counting conditions "
            "fail."
        ),
    )
    if feasible and exists is None:
        out.warnings.append(
            "The counting conditions and Fisher's inequality are NECESSARY, "
            "not sufficient. They passing does not establish that the design "
            "exists -- (22, 7, 2) passes all of them and is impossible by "
            "Bruck-Ryser-Chowla. `exists` is left undetermined here."
        )
    if exists is False and feasible:
        out.warnings.append(
            f"This parameter set is arithmetically feasible but the design "
            f"does not exist: {note}."
        )
    return out


def steiner_triple_system(v, construct=True):
    r"""A Steiner triple system exists exactly when
    :math:`v \equiv 1` or :math:`3 \pmod 6`.

    An :math:`\mathrm{STS}(v)` is a :math:`(v, 3, 1)`-BIBD: every pair
    of points lies in exactly one triple. Kirkman settled existence in
    1847, and unlike the general BIBD case the condition here is both
    necessary *and* sufficient -- which is why this function can return
    a definite answer where :func:`bibd_parameters` cannot.

    When ``construct`` and :math:`v \equiv 3 \pmod 6`, the Bose
    construction is carried out and the result verified: every pair
    covered exactly once.

    Examples
    --------
    >>> steiner_triple_system(7)["exists"]
    True
    >>> steiner_triple_system(8)["exists"]
    False
    >>> out = steiner_triple_system(9)
    >>> out["n_triples"], out["verified"]
    (12, True)
    """
    v = int(v)
    if v < 0:
        raise ValueError(f"v must be non-negative; got {v}.")
    exists = v in (0, 1) or v % 6 in (1, 3)
    n_triples = v * (v - 1) // 6 if exists else None

    triples = None
    verified = None
    if construct and exists and v >= 3 and v % 6 == 3:
        n = v // 3
        # Bose construction over Z_n x {0,1,2} with n odd
        idx = {}
        c = 0
        for i in range(n):
            for j in range(3):
                idx[(i, j)] = c
                c += 1
        half = (n + 1) // 2
        triples = []
        for i in range(n):
            triples.append([idx[(i, 0)], idx[(i, 1)], idx[(i, 2)]])
        for j in range(3):
            for a in range(n):
                for b in range(a + 1, n):
                    m = ((a + b) * half) % n
                    triples.append([idx[(a, j)], idx[(b, j)],
                                    idx[(m, (j + 1) % 3)]])
        seen = {}
        ok = True
        for t in triples:
            for p in combinations(sorted(t), 2):
                seen[p] = seen.get(p, 0) + 1
        verified = (len(triples) == n_triples
                    and len(seen) == math.comb(v, 2)
                    and all(x == 1 for x in seen.values()))

    out = RichResult(
        title=f"Steiner triple system STS({v})",
        summary_lines=[
            ("Exists", exists),
            ("v mod 6", v % 6 if v else 0),
            ("Number of triples", n_triples),
            ("Construction verified", verified),
        ],
        payload={
            "v": v,
            "exists": exists,
            "estimate": float(n_triples) if n_triples is not None
            else float("nan"),
            "n_triples": n_triples,
            "triples": triples,
            "verified": verified,
            "condition": "v = 1 or 3 (mod 6)",
            "condition_is_sufficient": True,
            "n": v,
            "method": "Steiner triple system (Kirkman 1847; Bose 1939)",
        },
        interpretation=(
            f"STS({v}) exists and has {n_triples} triples."
            if exists else
            f"No STS({v}) exists: {v} is {v % 6} mod 6, and only 1 and 3 "
            "admit one."
        ),
    )
    if verified is False:
        out.warnings.append(
            "The Bose construction did not produce a valid system: some pair "
            "is covered a number of times other than once. The construction "
            "is wrong, not the existence theorem."
        )
    return out


def latin_square(n, method="cyclic"):
    r"""A Latin square of order :math:`n`.

    The cyclic construction :math:`L[i][j] = (i + j) \bmod n` works for
    every :math:`n`, so Latin squares -- unlike most designs -- exist
    unconditionally.

    Examples
    --------
    >>> L = latin_square(4)["square"]
    >>> is_latin_square(L)["valid"]
    True
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    if method == "cyclic":
        L = [[(i + j) % n for j in range(n)] for i in range(n)]
    elif method == "shifted":
        L = [[(i * 2 + j) % n for j in range(n)] for i in range(n)]
    else:
        raise ValueError('method must be "cyclic" or "shifted".')
    chk = is_latin_square(L)
    return {"square": L, "order": n, "valid": chk["valid"], "method": method}


def is_latin_square(square):
    """Is every symbol used exactly once in each row and each column?"""
    L = [list(r) for r in square]
    n = len(L)
    if n == 0:
        raise ValueError("the square must not be empty.")
    if any(len(r) != n for r in L):
        raise ValueError("the square must be square.")
    syms = set()
    for r in L:
        syms.update(r)
    ok_rows = all(len(set(r)) == n for r in L)
    ok_cols = all(len({L[i][j] for i in range(n)}) == n for j in range(n))
    ok_syms = len(syms) == n
    return {"valid": ok_rows and ok_cols and ok_syms,
            "rows_ok": ok_rows, "columns_ok": ok_cols,
            "symbol_count_ok": ok_syms, "order": n, "symbols": sorted(syms)}


def are_orthogonal(square_a, square_b):
    r"""Are two Latin squares orthogonal (a Graeco-Latin pair)?

    Orthogonal means every ordered pair of symbols occurs exactly once
    when the squares are superimposed, so the :math:`n^2` cells produce
    all :math:`n^2` pairs.

    Euler conjectured in 1782 that no pair exists for
    :math:`n \equiv 2 \pmod 4`. He was right at :math:`n = 2` and
    :math:`n = 6` -- the latter proved by Tarry in 1901 by exhaustion --
    and **wrong for every larger such** :math:`n`, as Bose, Shrikhande
    and Parker showed in 1960 by constructing a pair of order 10. The
    conjecture stood for 178 years.

    Returns
    -------
    dict with ``orthogonal``, ``pairs_seen``, ``pairs_needed``,
    ``missing``.
    """
    A = [list(r) for r in square_a]
    B = [list(r) for r in square_b]
    n = len(A)
    if len(B) != n:
        raise ValueError("the two squares must have the same order.")
    if any(len(r) != n for r in A) or any(len(r) != n for r in B):
        raise ValueError("both squares must be square.")
    # orthogonality is a statement ABOUT Latin squares; checking the
    # pair condition on a grid that is not one gives a number with no
    # meaning, so both are validated first
    va = is_latin_square(A)["valid"]
    vb = is_latin_square(B)["valid"]
    pairs = {}
    for i in range(n):
        for j in range(n):
            p = (A[i][j], B[i][j])
            pairs[p] = pairs.get(p, 0) + 1
    cond = len(pairs) == n * n and all(c == 1 for c in pairs.values())
    return {"orthogonal": bool(cond and va and vb),
            "pair_condition_holds": cond,
            "both_are_latin": va and vb,
            "first_is_latin": va, "second_is_latin": vb,
            "pairs_seen": len(pairs), "pairs_needed": n * n,
            "repeated": sorted(p for p, c in pairs.items() if c > 1),
            "order": n}


def hamming_bound(n, d, q=2):
    r"""The sphere-packing bound on the size of a :math:`q`-ary code.

    A code correcting :math:`t = \lfloor (d-1)/2 \rfloor` errors has
    disjoint Hamming balls of radius :math:`t` around its codewords, so

    .. math::
        |C| \le \frac{q^n}{\sum_{i=0}^{t} \binom{n}{i} (q-1)^i}.

    A code meeting the bound with equality is **perfect**, and perfect
    codes are almost nonexistent: over any prime power alphabet the only
    ones are the trivial codes, the Hamming codes, and the two Golay
    codes. ``is_perfect_possible`` reports whether the bound is even an
    integer, which is a necessary condition for meeting it.

    Examples
    --------
    >>> out = hamming_bound(7, 3)          # the [7,4] Hamming code
    >>> out["bound"], out["is_perfect_possible"]
    (16, True)
    >>> hamming_bound(23, 7)["bound"]      # the binary Golay code
    4096
    """
    n, d, q = int(n), int(d), int(q)
    if n < 1 or d < 1 or q < 2:
        raise ValueError("n and d must be positive and q at least 2.")
    if d > n:
        raise ValueError(f"d must not exceed n; got d = {d}, n = {n}.")
    t = (d - 1) // 2
    vol = sum(math.comb(n, i) * (q - 1) ** i for i in range(t + 1))
    total = q ** n
    bound = total // vol
    exact = total % vol == 0
    return RichResult(
        title=f"Hamming bound, n = {n}, d = {d}, q = {q}",
        summary_lines=[
            ("Bound on |C|", bound),
            ("Correctable errors t", t),
            ("Ball volume", vol),
            ("Bound is exact", exact),
        ],
        payload={
            "bound": bound,
            "estimate": float(bound),
            "exact": str(bound),
            "errors_corrected": t,
            "ball_volume": vol,
            "total_words": total,
            "is_perfect_possible": exact,
            "rate_bound": (math.log(bound, q) / n if bound > 0 else
                           float("-inf")),
            "n": n, "d": d, "q": q,
            "method": "Hamming sphere-packing bound (Hamming 1950)",
        },
        interpretation=(
            f"No q-ary code of length {n} with minimum distance {d} has more "
            f"than {bound} codewords."
        ),
    )


def singleton_bound(n, d, q=2):
    r"""The Singleton bound :math:`|C| \le q^{n-d+1}`.

    Codes meeting it are *maximum distance separable*. Unlike perfect
    codes, MDS codes are plentiful -- Reed-Solomon codes are MDS for
    every length up to the alphabet size.

    Examples
    --------
    >>> singleton_bound(7, 3)["bound"]
    32
    """
    n, d, q = int(n), int(d), int(q)
    if n < 1 or d < 1 or q < 2:
        raise ValueError("n and d must be positive and q at least 2.")
    if d > n:
        raise ValueError(f"d must not exceed n; got d = {d}, n = {n}.")
    bound = q ** (n - d + 1)
    ham = hamming_bound(n, d, q)["bound"]
    return RichResult(
        title=f"Singleton bound, n = {n}, d = {d}, q = {q}",
        summary_lines=[
            ("Bound on |C|", bound),
            ("Hamming bound", ham),
            ("Tighter bound", min(bound, ham)),
        ],
        payload={
            "bound": bound,
            "estimate": float(bound),
            "exact": str(bound),
            "hamming_bound": ham,
            "tighter": min(bound, ham),
            "hamming_is_tighter": ham < bound,
            "n": n, "d": d, "q": q,
            "method": "Singleton bound (Singleton 1964)",
        },
    )


def incidence_matrix_check(blocks, v):
    r"""Verify a block design directly from its blocks.

    Counts how many blocks contain each point and each pair, and
    reports whether the design is balanced. This is the check that a
    claimed design *is* one, as opposed to having parameters that would
    permit one.

    Returns
    -------
    RichResult with ``is_bibd``, ``r``, ``k``, ``lambda``,
    ``pair_counts``, ``uncovered_pairs``.
    """
    B = [sorted(set(map(int, b))) for b in blocks]
    v = int(v)
    if v < 1:
        raise ValueError(f"v must be positive; got {v}.")
    if any(p < 0 or p >= v for b in B for p in b):
        raise ValueError(f"every point must lie in 0 .. {v - 1}.")
    sizes = {len(b) for b in B}
    point = [0] * v
    for b in B:
        for p in b:
            point[p] += 1
    pair = {}
    for b in B:
        for p in combinations(b, 2):
            pair[p] = pair.get(p, 0) + 1
    all_pairs = list(combinations(range(v), 2))
    uncovered = [p for p in all_pairs if p not in pair]
    counts = set(pair.values()) | ({0} if uncovered else set())
    reps = set(point)
    is_bibd = (len(sizes) == 1 and len(reps) == 1 and len(counts) == 1
               and not uncovered)
    # read these ONCE. An earlier version called sizes.pop() while
    # building the summary, which mutates the set, so the payload then
    # read an empty one and reported k = None on a valid Fano plane.
    k_val = next(iter(sizes)) if len(sizes) == 1 else None
    r_val = next(iter(reps)) if len(reps) == 1 else None
    lam_val = next(iter(counts)) if len(counts) == 1 else None
    out = RichResult(
        title="Block design check",
        summary_lines=[
            ("Is a BIBD", is_bibd),
            ("Blocks", len(B)),
            ("Block size k", k_val if k_val is not None else sorted(sizes)),
            ("Replication r", r_val if r_val is not None else sorted(reps)),
            ("lambda", lam_val if lam_val is not None else sorted(counts)),
        ],
        payload={
            "is_bibd": is_bibd,
            "estimate": float(len(B)),
            "b": len(B),
            "v": v,
            "k": k_val,
            "r": r_val,
            "lambda": lam_val,
            "block_sizes": sorted(sizes) if sizes else [],
            "point_replications": point,
            "pair_counts": pair,
            "uncovered_pairs": uncovered,
            "n": v,
            "method": "Direct verification of a block design",
        },
    )
    if uncovered:
        out.warnings.append(
            f"{len(uncovered)} pairs appear in no block, so this is not a "
            "design covering every pair."
        )
    if len(sizes) > 1:
        out.warnings.append(
            f"Blocks have differing sizes {sorted(sizes)}; a BIBD needs them "
            "uniform."
        )
    return out


def cheatsheet():
    return (
        "dsgnth: BIBD feasibility with Fisher, Steiner triple systems with "
        "the Bose construction verified, Latin squares and orthogonality, "
        "and the Hamming and Singleton coding bounds"
    )
