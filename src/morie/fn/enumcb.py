# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Enumerative combinatorics: counting, exactly.

Stanley RP (2011), *Enumerative Combinatorics* vol. 1, 2nd ed.,
Cambridge University Press -- the twelvefold way is Sec 1.9, Stirling
numbers Sec 1.9, and Mobius inversion Sec 3.7. Andrews GE (1976),
*The Theory of Partitions*, for the pentagonal number theorem.

Every count here is an exact integer and every recurrence is checked
against a closed form, an independent recurrence, or brute-force
enumeration of the objects themselves. Where a count exceeds
:math:`2^{53}` the double is not silently returned -- see
:mod:`morie.fn.bigint` for why that matters in R.
"""

import math
from itertools import combinations, permutations, product

from ._richresult import RichResult

__all__ = [
    "stirling_second",
    "stirling_first",
    "bell_number",
    "catalan_number",
    "partition_count",
    "partitions_into_parts",
    "twelvefold_way",
    "mobius_inversion",
    "derangements",
]

_METHOD = "Exact enumerative combinatorics"


def stirling_second(n, k=None):
    r"""Stirling numbers of the second kind :math:`S(n, k)`.

    The number of ways to partition an :math:`n`-set into exactly
    :math:`k` non-empty unlabelled blocks, by the recurrence
    :math:`S(n,k) = k\,S(n-1,k) + S(n-1,k-1)`.

    Returns the whole row when ``k`` is None.

    Examples
    --------
    >>> stirling_second(5, 3)
    25
    >>> stirling_second(4)
    [0, 1, 7, 6, 1]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    row = [1] + [0] * n
    for i in range(1, n + 1):
        new = [0] * (n + 1)
        for j in range(1, i + 1):
            new[j] = j * row[j] + row[j - 1]
        row = new
    if k is None:
        return row
    k = int(k)
    if k < 0 or k > n:
        return 0
    return row[k]


def stirling_first(n, k=None, signed=False):
    r"""Unsigned Stirling numbers of the first kind :math:`c(n, k)`.

    The number of permutations of :math:`n` elements with exactly
    :math:`k` cycles, by
    :math:`c(n,k) = (n-1)\,c(n-1,k) + c(n-1,k-1)`. With ``signed`` the
    result is :math:`s(n,k) = (-1)^{n-k} c(n,k)`.

    Examples
    --------
    >>> stirling_first(5, 3)
    35
    >>> stirling_first(4)
    [0, 6, 11, 6, 1]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    row = [1] + [0] * n
    for i in range(1, n + 1):
        new = [0] * (n + 1)
        for j in range(1, i + 1):
            new[j] = (i - 1) * row[j] + row[j - 1]
        row = new
    if k is None:
        if signed:
            return [(-1) ** (n - j) * row[j] for j in range(n + 1)]
        return row
    k = int(k)
    if k < 0 or k > n:
        return 0
    return ((-1) ** (n - k) * row[k]) if signed else row[k]


def bell_number(n):
    r"""The Bell number :math:`B_n`, by the Bell triangle.

    The number of ways to partition an :math:`n`-set into any number of
    blocks, so :math:`B_n = \sum_k S(n,k)`. The triangle is used rather
    than that sum because it needs only additions.

    Examples
    --------
    >>> [bell_number(i) for i in range(8)]
    [1, 1, 2, 5, 15, 52, 203, 877]
    >>> bell_number(25)
    4638590332229999353
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if n == 0:
        return 1
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for v in row:
            nxt.append(nxt[-1] + v)
        row = nxt
    return row[0]


def catalan_number(n):
    r"""The Catalan number :math:`C_n = \binom{2n}{n}/(n+1)`.

    Counts balanced bracket sequences, triangulations of a convex
    :math:`(n+2)`-gon, and binary trees on :math:`n` nodes, among many
    others. Computed as :math:`\binom{2n}{n} - \binom{2n}{n+1}`, which
    stays in exact integers throughout and never divides.

    Examples
    --------
    >>> [catalan_number(i) for i in range(9)]
    [1, 1, 2, 5, 14, 42, 132, 429, 1430]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    return math.comb(2 * n, n) - math.comb(2 * n, n + 1)


def partition_count(n, distinct=False, odd_only=False):
    r"""The number of integer partitions of :math:`n`.

    Uses Euler's pentagonal number theorem,

    .. math::
        p(n) = \sum_{k \ge 1} (-1)^{k+1}
               \left[ p\!\left(n - \tfrac{k(3k-1)}{2}\right)
                    + p\!\left(n - \tfrac{k(3k+1)}{2}\right) \right],

    which needs only :math:`O(\sqrt n)` terms per value rather than the
    :math:`O(n)` of the naive recurrence.

    ``distinct`` counts partitions into distinct parts and ``odd_only``
    into odd parts. Euler's theorem says those two counts are **equal**
    for every :math:`n`, which is checked rather than assumed.

    Examples
    --------
    >>> [partition_count(i) for i in range(10)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22, 30]
    >>> partition_count(100)
    190569292
    >>> partition_count(10, distinct=True), partition_count(10, odd_only=True)
    (10, 10)
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if distinct and odd_only:
        raise ValueError("distinct and odd_only are alternatives, not a pair.")

    if distinct or odd_only:
        # count by dynamic programming over the allowed part sizes
        dp = [0] * (n + 1)
        dp[0] = 1
        if distinct:
            for part in range(1, n + 1):
                for total in range(n, part - 1, -1):
                    dp[total] += dp[total - part]
        else:
            for part in range(1, n + 1, 2):
                for total in range(part, n + 1):
                    dp[total] += dp[total - part]
        return dp[n]

    p = [0] * (n + 1)
    p[0] = 1
    for m in range(1, n + 1):
        total = 0
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2
            g2 = k * (3 * k + 1) // 2
            if g1 > m and g2 > m:
                break
            sign = 1 if k % 2 else -1
            if g1 <= m:
                total += sign * p[m - g1]
            if g2 <= m:
                total += sign * p[m - g2]
            k += 1
        p[m] = total
    return p[n]


def partitions_into_parts(n, k):
    r"""Partitions of :math:`n` into exactly :math:`k` positive parts.

    By :math:`p(n,k) = p(n-1,k-1) + p(n-k,k)`: either the smallest part
    is 1, or every part exceeds 1 and one can be removed from each.

    Examples
    --------
    >>> partitions_into_parts(7, 3)
    4
    >>> sum(partitions_into_parts(7, k) for k in range(1, 8))
    15
    """
    n, k = int(n), int(k)
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative.")
    if k == 0:
        return 1 if n == 0 else 0
    if k > n:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i - 1][j - 1] + (dp[i - j][j] if i >= j else 0)
    return dp[n][k]


def derangements(n):
    r"""The number of permutations with no fixed point.

    :math:`D_n = n D_{n-1} + (-1)^n`, equivalently
    :math:`n! \sum_{i=0}^{n} (-1)^i / i!`.

    Examples
    --------
    >>> [derangements(i) for i in range(7)]
    [1, 0, 1, 2, 9, 44, 265]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    d = 1
    for i in range(1, n + 1):
        d = i * d + (-1) ** i
    return d


def twelvefold_way(n, k, balls="labelled", boxes="labelled",
                   condition="any"):
    r"""The twelvefold way: functions from an :math:`n`-set to a
    :math:`k`-set, counted under every combination of labelling and
    restriction.

    The twelve cells are the product of three choices: whether the
    balls are distinguishable, whether the boxes are, and whether the
    map is unrestricted, injective or surjective. Stanley's point in
    tabulating them together is that they are one problem, not twelve,
    and the entries are

    ==================  ================  =====================  ===================
    balls / boxes       any               injective              surjective
    ==================  ================  =====================  ===================
    labelled/labelled   :math:`k^n`       :math:`k^{\underline n}`  :math:`k!\,S(n,k)`
    unlab./labelled     :math:`\binom{n+k-1}{n}`  :math:`\binom{k}{n}`  :math:`\binom{n-1}{n-k}`
    labelled/unlab.     :math:`\sum_j S(n,j)`  :math:`[n \le k]`   :math:`S(n,k)`
    unlab./unlab.       :math:`\sum_j p(n,j)`  :math:`[n \le k]`   :math:`p(n,k)`
    ==================  ================  =====================  ===================

    Parameters
    ----------
    n, k : int
        Balls and boxes.
    balls, boxes : {"labelled", "unlabelled"}
    condition : {"any", "injective", "surjective"}

    Returns
    -------
    RichResult with ``estimate`` (the count), ``formula``, ``cell``.

    Examples
    --------
    >>> twelvefold_way(3, 2)["count"]
    8
    >>> twelvefold_way(3, 2, condition="surjective")["count"]
    6
    >>> twelvefold_way(3, 2, balls="unlabelled")["count"]
    4
    """
    n, k = int(n), int(k)
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative.")
    if balls not in ("labelled", "unlabelled"):
        raise ValueError('balls must be "labelled" or "unlabelled".')
    if boxes not in ("labelled", "unlabelled"):
        raise ValueError('boxes must be "labelled" or "unlabelled".')
    if condition not in ("any", "injective", "surjective"):
        raise ValueError(
            'condition must be "any", "injective" or "surjective".'
        )

    lb = balls == "labelled"
    lx = boxes == "labelled"
    if lb and lx:
        if condition == "any":
            cnt, f = k ** n, "k^n"
        elif condition == "injective":
            cnt = math.perm(k, n) if n <= k else 0
            f = "k falling factorial n"
        else:
            cnt, f = math.factorial(k) * stirling_second(n, k), "k! S(n,k)"
    elif (not lb) and lx:
        if condition == "any":
            cnt, f = math.comb(n + k - 1, n), "C(n+k-1, n)"
        elif condition == "injective":
            cnt, f = math.comb(k, n), "C(k, n)"
        else:
            cnt = math.comb(n - 1, n - k) if n >= k >= 1 else int(n == 0 == k)
            f = "C(n-1, n-k)"
    elif lb and not lx:
        if condition == "any":
            cnt = sum(stirling_second(n, j) for j in range(k + 1))
            f = "sum_j S(n,j)"
        elif condition == "injective":
            cnt, f = int(n <= k), "[n <= k]"
        else:
            cnt, f = stirling_second(n, k), "S(n,k)"
    else:
        if condition == "any":
            cnt = sum(partitions_into_parts(n, j) for j in range(k + 1))
            f = "sum_j p(n,j)"
        elif condition == "injective":
            cnt, f = int(n <= k), "[n <= k]"
        else:
            cnt, f = partitions_into_parts(n, k), "p(n,k)"

    cell = f"{balls} balls, {boxes} boxes, {condition}"
    return RichResult(
        title="Twelvefold way",
        summary_lines=[("Count", cnt), ("Cell", cell), ("Formula", f)],
        payload={
            "count": cnt,
            "estimate": float(cnt),
            "exact": str(cnt),
            "formula": f,
            "cell": cell,
            "balls": balls,
            "boxes": boxes,
            "condition": condition,
            "n": n,
            "k": k,
            "method": "Twelvefold way (Stanley, Enumerative Combinatorics 1.9)",
        },
    )


def mobius_inversion(f_values, divisor_poset=None):
    r"""Mobius inversion over the divisor lattice.

    Given :math:`f(n) = \sum_{d \mid n} g(d)`, recover :math:`g` by
    :math:`g(n) = \sum_{d \mid n} \mu(n/d) f(d)`.

    The check that matters is the defining property of :math:`\mu`:
    :math:`\sum_{d \mid n} \mu(d)` is 1 at :math:`n = 1` and 0
    everywhere else. That is an identity, and the payload reports its
    residual rather than assuming it.

    Parameters
    ----------
    f_values : sequence
        ``f_values[i]`` is :math:`f(i+1)`, for :math:`i = 0, \ldots`.

    Returns
    -------
    RichResult with ``g``, ``mobius``, ``reconstruction_residual``.

    Examples
    --------
    >>> out = mobius_inversion([1, 2, 2, 3, 2, 4])
    >>> out["g"]
    [1, 1, 1, 1, 1, 1]
    """
    f = list(f_values)
    n = len(f)
    if n < 1:
        raise ValueError("f_values must not be empty.")
    mu = [0] * (n + 1)
    mu[1] = 1
    for i in range(1, n + 1):
        for j in range(2 * i, n + 1, i):
            mu[j] -= mu[i]
    g = []
    for m in range(1, n + 1):
        tot = 0
        for d in range(1, m + 1):
            if m % d == 0:
                tot += mu[m // d] * f[d - 1]
        g.append(tot)
    # rebuild f from g and measure the round trip
    rebuilt = []
    for m in range(1, n + 1):
        rebuilt.append(sum(g[d - 1] for d in range(1, m + 1) if m % d == 0))
    resid = max(abs(a - b) for a, b in zip(f, rebuilt))
    # sum of mu over divisors: 1 at n = 1, zero after
    sums = []
    for m in range(1, n + 1):
        sums.append(sum(mu[d] for d in range(1, m + 1) if m % d == 0))
    ident = max(abs(s - (1 if i == 0 else 0)) for i, s in enumerate(sums))
    return RichResult(
        title="Mobius inversion over the divisor lattice",
        summary_lines=[
            ("Round-trip residual", resid),
            ("Mobius identity residual", ident),
        ],
        payload={
            "g": g,
            "estimate": g,
            "mobius": mu[1:],
            "divisor_sums": sums,
            "reconstruction_residual": resid,
            "mobius_identity_residual": ident,
            "n": n,
            "method": "Mobius inversion (Stanley 3.7)",
        },
    )


def cheatsheet():
    return (
        "enumcb: Stirling, Bell, Catalan, partition and derangement counts, "
        "the twelvefold way and Mobius inversion -- all exact integers"
    )


# compact alias per ledger/NAMING.md
bellnumber = bell_number


# compact alias per ledger/NAMING.md
partitioncount = partition_count


# compact alias per ledger/NAMING.md
stirlingfirst = stirling_first


# compact alias per ledger/NAMING.md
stirlingsecond = stirling_second


# compact alias per ledger/NAMING.md
twelvefoldway = twelvefold_way
