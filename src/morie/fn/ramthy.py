# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ramsey theory: guaranteed monochromatic structure in large graphs.

This is F. P. Ramsey's combinatorics -- complete disorder is
impossible -- and has nothing to do with J. B. Ramsey's RESET
specification test in :mod:`morie.fn.rsetf`, which is named after a
different person and tests a different thing entirely.

Ramsey F. P. (1930), *On a problem of formal logic*, Proceedings of
the London Mathematical Society s2-30:264-286. Values and bounds
follow Radziszowski S. P., *Small Ramsey Numbers*, Electronic Journal
of Combinatorics, Dynamic Survey DS1, revision 17 (2024),
doi:10.37236/21, Tables Ia and Ib. Goodman A. W. (1959), *On sets of
acquaintances and strangers at any party*, American Mathematical
Monthly 66(9):778-783. Erdos P. (1947), *Some remarks on the theory
of graphs*, Bulletin of the AMS 53:292-294.
"""

import math
from itertools import combinations

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "ramsey_number",
    "monochromatic_triangles",
    "goodman_minimum",
    "ramsey_upper_bound",
    "ramsey_lower_bound_probabilistic",
    "verify_ramsey_witness",
    "party_problem",
]

_METHOD = "Ramsey number lookup with certified bounds"

# Exactly known nontrivial two-colour values, DS1 revision 17 Table Ia.
# Only nine are known for k, l >= 3 -- that is the entire list, and the
# reason larger ones are quoted as intervals rather than numbers.
_KNOWN = {
    (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23,
    (3, 8): 28, (3, 9): 36, (4, 4): 18, (4, 5): 25,
}

# Best published bounds for a few unknown cases, DS1 Tables Ia and Ib.
# Upper bounds for k >= 4 are the 2023 Angeltveit-McKay values.
_BOUNDS = {
    (3, 10): (40, 41), (3, 11): (47, 50), (3, 12): (53, 59),
    (3, 13): (60, 68), (4, 6): (36, 40), (4, 7): (49, 58),
    (4, 8): (59, 79), (5, 5): (43, 46), (5, 6): (59, 85),
    (6, 6): (102, 160),
}

_CREDITS = {
    (3, 3): "Kurschak (1947); Putnam problem (1953)",
    (3, 4): "Greenwood and Gleason (1955)",
    (3, 5): "Greenwood and Gleason (1955)",
    (3, 6): "Kery (1964)",
    (4, 4): "Greenwood and Gleason (1955)",
    (4, 5): "McKay and Radziszowski (1995); HOL4-verified",
    (3, 8): "Grinstead and Roberts (1982); DRAT-verified",
}


def ramsey_number(k, l=None):
    r"""The Ramsey number :math:`R(k, l)`, exactly or as an interval.

    :math:`R(k, l)` is the least :math:`n` such that every red-blue
    colouring of the edges of :math:`K_n` contains a red :math:`K_k` or
    a blue :math:`K_l`.

    **Almost nothing is known.** For :math:`k, l \ge 3` exactly nine
    values have ever been determined, and the list has not grown since
    1995. :math:`R(5,5)` is unknown: it lies in :math:`[43, 46]`, and
    the lower end is conjectured to be the answer. The difficulty is
    not that the problem is open-ended but that it is finite and
    enormous -- deciding :math:`R(5,5)` by exhaustive search means
    examining colourings of :math:`K_{43}`, of which there are
    :math:`2^{903}`.

    This function returns a value only when one is known, and an
    interval otherwise. It never interpolates, and it never returns a
    bound as though it were a value.

    A specific caution encoded here: the claim :math:`R(5,5) = 50`
    circulates and is **wrong**. DS1 notes it has been shown incorrect
    more than once and is still cited. Asking for ``ramsey_number(5, 5)``
    returns the interval and says so.

    Parameters
    ----------
    k, l : int
        Clique sizes. ``ramsey_number(k)`` means the diagonal
        :math:`R(k, k)`.

    Returns
    -------
    RichResult
        ``value`` (None when unknown), ``lower``, ``upper``, ``exact``,
        ``credit``, ``erdos_szekeres_bound``.

    References
    ----------
    Radziszowski SP, *Small Ramsey Numbers*, EJC Dynamic Survey DS1,
    rev. 17 (2024), doi:10.37236/21, Tables Ia and Ib.

    Examples
    --------
    >>> ramsey_number(3, 3)["value"]
    6
    >>> ramsey_number(4, 4)["value"]
    18
    >>> ramsey_number(5, 5)["value"] is None
    True
    >>> ramsey_number(5, 5)["lower"], ramsey_number(5, 5)["upper"]
    (43, 46)
    """
    if l is None:
        l = k
    k, l = int(k), int(l)
    if k < 1 or l < 1:
        raise ValueError(f"k and l must be at least 1; got {k}, {l}.")
    a, b = min(k, l), max(k, l)

    # trivial cases, exact by definition
    if a == 1:
        val, lo, hi, credit = 1, 1, 1, "trivial: R(1, l) = 1"
    elif a == 2:
        val, lo, hi, credit = b, b, b, "trivial: R(2, l) = l"
    elif (a, b) in _KNOWN:
        val = _KNOWN[(a, b)]
        lo = hi = val
        credit = _CREDITS.get((a, b), "DS1 Table Ia")
    elif (a, b) in _BOUNDS:
        val = None
        lo, hi = _BOUNDS[(a, b)]
        credit = "DS1 Tables Ia and Ib (bounds only)"
    else:
        val = None
        lo = None
        hi = _erdos_szekeres(a, b)
        credit = "no published bound tabulated here"

    es = _erdos_szekeres(a, b)
    out = RichResult(
        title=f"Ramsey number R({k}, {l})",
        summary_lines=[
            ("R(k, l)", val if val is not None else f"[{lo}, {hi}]"),
            ("Exactly known", val is not None),
            ("Erdos-Szekeres upper bound", es),
        ],
        payload={
            "k": k, "l": l,
            "value": val,
            "estimate": float(val) if val is not None else float("nan"),
            "lower": lo,
            "upper": hi,
            "exact": val is not None,
            "credit": credit,
            "erdos_szekeres_bound": es,
            "symmetric": True,
            "n": val if val is not None else hi,
            "method": _METHOD,
        },
        interpretation=(
            f"R({k}, {l}) = {val}."
            if val is not None else
            f"R({k}, {l}) is not known; it lies in [{lo}, {hi}]."
            if lo is not None else
            f"R({k}, {l}) is not known and no tabulated lower bound is "
            f"carried here; Erdos-Szekeres gives R <= {es}."
        ),
    )
    if val is None:
        out.warnings.append(
            f"R({k}, {l}) has never been determined. The interval is a pair "
            "of proofs, not an estimate, and the true value is not more "
            "likely to sit in the middle of it."
        )
    if (a, b) == (5, 5):
        out.warnings.append(
            "The frequently repeated claim that R(5,5) = 50 is incorrect. "
            "DS1 records that it has been shown wrong more than once and is "
            "still being cited. The published interval is [43, 46], with 43 "
            "conjectured."
        )
    return out


def _erdos_szekeres(k, l):
    r"""The bound :math:`R(k,l) \le \binom{k+l-2}{k-1}`."""
    return int(math.comb(k + l - 2, k - 1))


def ramsey_upper_bound(k, l, use_known=True):
    r"""Upper bounds on :math:`R(k, l)` from the two classical arguments.

    The recursive bound is
    :math:`R(k,l) \le R(k-1,l) + R(k,l-1)`, with the inequality
    **strict** when both terms on the right are even. Unrolling it with
    :math:`R(2,l) = l` gives the Erdos-Szekeres binomial bound
    :math:`\binom{k+l-2}{k-1}`.

    The binomial form is the one usually quoted, and it is very weak:
    it gives :math:`R(4,4) \le 20` against the true 18, and
    :math:`R(5,5) \le 70` against a true value of at most 46. Both are
    reported so the gap is visible.

    ``use_known`` decides whether the recursion may short-circuit on
    the nine tabulated values. It is on by default because that gives
    the tightest bound, but it makes the result **partly a lookup**:
    with it on, the recursion "derives" R(4,5) <= 25 only because 25
    was fed in. Setting it False gives the bound the argument alone
    supports, and the two differ -- 27 against 25 for R(4,5). Any claim
    that the recursion reproduces a known value should use False.

    Returns
    -------
    dict with ``binomial``, ``recursive``, ``best``, ``parity_saving``,
    ``used_known_values``.
    """
    k, l = int(k), int(l)
    if k < 1 or l < 1:
        raise ValueError(f"k and l must be at least 1; got {k}, {l}.")

    memo = {}

    def rec(a, b):
        a, b = min(a, b), max(a, b)
        if a == 1:
            return 1
        if a == 2:
            return b
        if use_known and (a, b) in _KNOWN:
            return _KNOWN[(a, b)]
        if (a, b) in memo:
            return memo[(a, b)]
        u1, u2 = rec(a - 1, b), rec(a, b - 1)
        val = u1 + u2
        # Greenwood-Gleason: strict when both are even
        if u1 % 2 == 0 and u2 % 2 == 0:
            val -= 1
        memo[(a, b)] = val
        return val

    r = rec(k, l)
    binom = _erdos_szekeres(min(k, l), max(k, l))
    return {"binomial": binom, "recursive": r, "best": min(binom, r),
            "parity_saving": binom - r, "used_known_values": bool(use_known)}


def ramsey_lower_bound_probabilistic(k):
    r"""Erdos's probabilistic lower bound on the diagonal :math:`R(k,k)`.

    Colour each edge of :math:`K_n` independently at random. The
    expected number of monochromatic :math:`K_k` is
    :math:`\binom{n}{k} 2^{1 - \binom{k}{2}}`. If that is below 1 then
    some colouring has none, so :math:`R(k,k) > n`.

    The argument is famous for proving that an object exists without
    producing one, and seventy-five years later no explicit
    construction comes close to it. For :math:`k = 10` this certifies
    :math:`R(10,10) > 100` or so from a two-line calculation, while
    the best known explicit colouring is far weaker. The bound it
    implies asymptotically is :math:`R(k,k) > 2^{k/2}`.

    Returns
    -------
    dict with ``bound`` (largest certified n), ``expected_at_bound``,
    ``asymptotic_2_to_k_over_2``.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"k must be at least 2; got {k}.")
    # log2 E[# mono K_k] = log2 C(n,k) + 1 - C(k,2)
    exponent = 1 - math.comb(k, 2)
    n = k
    best = k - 1
    while n < 100000:
        log2_exp = (math.lgamma(n + 1) - math.lgamma(k + 1)
                    - math.lgamma(n - k + 1)) / math.log(2.0) + exponent
        if log2_exp < 0:
            best = n
            n += 1
        else:
            break
    log2_at = (math.lgamma(best + 1) - math.lgamma(k + 1)
               - math.lgamma(best - k + 1)) / math.log(2.0) + exponent
    return {
        "bound": best,
        "certifies": f"R({k},{k}) > {best}",
        "expected_at_bound": float(2.0 ** log2_at),
        "asymptotic_2_to_k_over_2": float(2.0 ** (k / 2.0)),
    }


def monochromatic_triangles(colouring, brute_force=False):
    r"""Count monochromatic triangles, via Goodman's identity.

    For any two-colouring of :math:`K_n`, a triangle fails to be
    monochromatic exactly when two of its vertices see one edge of each
    colour. Summing over vertices counts each such triangle twice, so

    .. math::
        \#\text{mono} = \binom{n}{3}
                        - \tfrac12 \sum_{v} r_v b_v,

    with :math:`r_v` and :math:`b_v` the red and blue degrees at
    :math:`v`. This is an **identity**, not an estimate: it holds
    exactly for every colouring, and ``brute_force=True`` checks it
    against direct enumeration.

    The identity is what makes :math:`R(3,3) \le 6` provable by hand.
    Since :math:`r_v + b_v = n - 1`, the product is at most
    :math:`\lfloor (n-1)^2/4 \rfloor`, so at :math:`n = 6` the count is
    at least :math:`20 - \tfrac12 \cdot 6 \cdot 6 = 2`. Every colouring
    of :math:`K_6` therefore contains not one monochromatic triangle
    but two. At :math:`n = 5` the same bound gives zero, and the
    five-cycle colouring attains it.

    Parameters
    ----------
    colouring : array-like, shape (n, n)
        Symmetric 0/1 matrix; 1 marks a "red" edge. The diagonal is
        ignored.
    brute_force : bool
        Also enumerate all triangles and check the identity.

    Returns
    -------
    RichResult with ``estimate`` (the count), ``red_triangles``,
    ``blue_triangles``, ``bichromatic``, ``goodman_minimum``,
    ``identity_residual``.

    Examples
    --------
    >>> import numpy as np
    >>> # the 5-cycle colouring of K5 has no monochromatic triangle
    >>> C = np.zeros((5, 5), dtype=int)
    >>> for i in range(5):
    ...     C[i, (i + 1) % 5] = C[(i + 1) % 5, i] = 1
    >>> monochromatic_triangles(C)["estimate"]
    0
    """
    A = np.atleast_2d(np.asarray(colouring))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"colouring must be square; got shape {A.shape}.")
    n = A.shape[0]
    if n < 3:
        raise ValueError(f"need at least three vertices; got {n}.")
    R = (np.asarray(A) != 0).astype(int)
    np.fill_diagonal(R, 0)
    if not np.array_equal(R, R.T):
        raise ValueError("colouring must be symmetric.")

    r = R.sum(axis=1)
    b = (n - 1) - r
    bichromatic_x2 = int(np.sum(r * b))
    if bichromatic_x2 % 2:
        raise ValueError(
            "the vertex sum is odd, which is impossible for a valid "
            "colouring; check the matrix is symmetric with a zero diagonal."
        )
    bichromatic = bichromatic_x2 // 2
    total = math.comb(n, 3)
    mono = total - bichromatic

    red_t = blue_t = None
    resid = None
    if brute_force:
        red_t = blue_t = 0
        for i, j, k in combinations(range(n), 3):
            s = R[i, j] + R[i, k] + R[j, k]
            if s == 3:
                red_t += 1
            elif s == 0:
                blue_t += 1
        resid = abs((red_t + blue_t) - mono)

    gmin = goodman_minimum(n)
    out = RichResult(
        title="Monochromatic triangles (Goodman's identity)",
        summary_lines=[
            ("Monochromatic", mono),
            ("Bichromatic", bichromatic),
            ("All triangles", total),
            ("Goodman minimum for n", gmin["minimum"]),
        ],
        payload={
            "estimate": mono,
            "monochromatic": mono,
            "bichromatic": bichromatic,
            "total_triangles": total,
            "red_triangles": red_t,
            "blue_triangles": blue_t,
            "identity_residual": resid,
            "goodman_minimum": gmin["minimum"],
            "attains_minimum": mono == gmin["minimum"],
            "red_degrees": r,
            "n": n,
            "method": "Goodman (1959) monochromatic-triangle identity",
        },
        interpretation=(
            f"This colouring of K_{n} has {mono} monochromatic triangle(s); "
            f"no colouring of K_{n} can have fewer than "
            f"{gmin['minimum']}."
        ),
    )
    if mono < gmin["minimum"]:
        out.warnings.append(
            f"The count {mono} is below Goodman's minimum {gmin['minimum']}, "
            "which is impossible. The colouring or the arithmetic is wrong."
        )
    return out


def goodman_minimum(n):
    r"""The least number of monochromatic triangles in any 2-colouring.

    From the identity in :func:`monochromatic_triangles`, maximising
    :math:`\sum_v r_v b_v` subject to :math:`r_v + b_v = n - 1` gives

    .. math::
        \#\text{mono} \ \ge\ \left\lceil \binom{n}{3}
        - \frac{n}{2}\left\lfloor \frac{(n-1)^2}{4}\right\rfloor
        \right\rceil .

    The ceiling matters: the bracket is often a half-integer and the
    count is an integer.

    Returns
    -------
    dict with ``minimum``, ``total_triangles``, ``max_bichromatic``.
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"n must be at least 3; got {n}.")
    total = math.comb(n, 3)
    max_bi_x2 = n * ((n - 1) ** 2 // 4)
    # the vertex sum must be even for a realisable colouring
    max_bi = max_bi_x2 // 2
    return {"minimum": max(total - max_bi, 0),
            "total_triangles": total,
            "max_bichromatic": max_bi}


def verify_ramsey_witness(colouring, k, l):
    """Check a colouring really avoids a red K_k and a blue K_l.

    A lower bound :math:`R(k,l) > n` is proved by exhibiting such a
    colouring of :math:`K_n`. This verifies the witness by exhaustive
    search over cliques, which is the only honest way to accept one.

    Returns
    -------
    dict with ``valid``, ``red_clique``, ``blue_clique``, ``n``.
    """
    A = np.atleast_2d(np.asarray(colouring))
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"colouring must be square; got shape {A.shape}.")
    R = (np.asarray(A) != 0).astype(int)
    np.fill_diagonal(R, 0)
    if not np.array_equal(R, R.T):
        raise ValueError("colouring must be symmetric.")
    k, l = int(k), int(l)

    red = None
    if k <= n:
        for c in combinations(range(n), k):
            if all(R[i, j] for i, j in combinations(c, 2)):
                red = list(c)
                break
    blue = None
    if l <= n:
        for c in combinations(range(n), l):
            if all(not R[i, j] for i, j in combinations(c, 2)):
                blue = list(c)
                break
    return {"valid": red is None and blue is None,
            "red_clique": red, "blue_clique": blue, "n": n,
            "certifies": (f"R({k},{l}) > {n}"
                          if red is None and blue is None else None)}


def party_problem(n_people=6):
    r"""The party problem: :math:`R(3,3) = 6`, proved rather than quoted.

    Among any six people, three are mutual acquaintances or three are
    mutual strangers. Five is not enough.

    Both halves are established here rather than asserted. The upper
    half follows from Goodman's identity, which forces at least two
    monochromatic triangles at :math:`n = 6`. The lower half needs a
    witness, and the five-cycle colouring is one: acquaintance around a
    pentagon, strangers on the diagonals. Neither the pentagon nor the
    pentagram contains a triangle.

    Returns
    -------
    RichResult with ``guaranteed``, ``minimum_monochromatic``,
    ``witness``, ``witness_valid``.

    Examples
    --------
    >>> out = party_problem(6)
    >>> out["guaranteed"]
    True
    >>> out["minimum_monochromatic"]
    2
    >>> party_problem(5)["guaranteed"]
    False
    """
    n = int(n_people)
    if n < 3:
        raise ValueError(f"n_people must be at least 3; got {n}.")
    gmin = goodman_minimum(n)["minimum"]
    guaranteed = gmin >= 1

    witness = None
    valid = None
    if not guaranteed:
        # the 5-cycle colouring, and its analogue for smaller n
        C = np.zeros((n, n), dtype=int)
        for i in range(n):
            C[i, (i + 1) % n] = C[(i + 1) % n, i] = 1
        witness = C
        valid = verify_ramsey_witness(C, 3, 3)["valid"]

    out = RichResult(
        title=f"The party problem at n = {n}",
        summary_lines=[
            ("Monochromatic triangle guaranteed", guaranteed),
            ("Goodman minimum", gmin),
            ("R(3,3)", 6),
        ],
        payload={
            "n_people": n,
            "guaranteed": guaranteed,
            "minimum_monochromatic": gmin,
            "estimate": float(gmin),
            "witness": witness,
            "witness_valid": valid,
            "ramsey_number": 6,
            "n": n,
            "method": "Party problem via Goodman's identity",
        },
        interpretation=(
            f"Among any {n} people there must be at least {gmin} "
            "monochromatic triangle(s): three mutual acquaintances or three "
            "mutual strangers."
            if guaranteed else
            f"With {n} people no monochromatic triangle is forced; the cycle "
            "colouring avoids one entirely."
        ),
    )
    return out


def cheatsheet():
    return (
        "ramthy: Ramsey numbers with certified bounds (DS1 rev 17), "
        "Goodman's monochromatic-triangle identity, the Erdos probabilistic "
        "lower bound, and witness verification. Not the RESET test."
    )


# compact alias per ledger/NAMING.md
goodmanminimum = goodman_minimum


# compact alias per ledger/NAMING.md
partyproblem = party_problem


# compact alias per ledger/NAMING.md
ramseynumber = ramsey_number
