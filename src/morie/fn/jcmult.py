# morie.fn -- function file (rootcoder007/morie)
"""Join-count statistics for a map of k colours (Cliff and Ord 1981)."""

from __future__ import annotations

from itertools import combinations

from . import _array_core as np
from ._richresult import RichResult
from .spcorr import _constants

__all__ = ["join_count_multi"]


def _pairs(v, op):
    # (i, j) for i = 1..k-1, j < i, in the order of spdep's nrns()
    return [op(v[i], v[j]) for i in range(1, len(v)) for j in range(i)]


def join_count_multi(labels, W) -> RichResult:
    r"""Same-colour and different-colour join counts for ``k >= 2`` colours.

    ``J_rr`` counts joins between two units of colour ``r`` and ``J_rs``
    (``r > s``) joins between colours ``r`` and ``s``, each as half the sum
    of the weights over ordered pairs; ``Jtot`` is the sum of all
    different-colour joins.  Expectations and variances are the nonfree
    sampling moments of Cliff and Ord (1981, pp. 19-20), exactly as
    ``spdep::joincount.multi`` (with ``adjust.n = TRUE``: ``N`` counts the
    units with neighbours), and the deviates are
    ``(J - E) / sqrt(Var)``.  Colours are the sorted distinct labels.

    :param labels: Colour of each unit (n,).
    :param W: Spatial weights (n, n); the diagonal is ignored.
    :return: :class:`RichResult` with ``rows`` (row names ``a:a``, ``b:a``,
        ..., ``Jtot``), ``joincount``, ``expected``, ``variance``, ``z``,
        ``levels``.
    :raises ValueError: With fewer than two colours or a bad ``W``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> W = [[1 if abs(i - j) == 1 else 0 for j in range(9)] for i in range(9)]
    >>> r = join_count_multi(list("aabbbccaa"), W)
    >>> r.rows
    ['a:a', 'b:b', 'c:c', 'b:a', 'c:a', 'c:b', 'Jtot']
    >>> r.joincount
    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0]
    """
    lab = list(labels)
    n = len(lab)
    Wl = np.asarray(W, dtype=float).tolist()
    if len(Wl) != n or any(len(r) != n for r in Wl):
        raise ValueError("W must be n x n with n = len(labels)")
    Wl = [[0.0 if i == j else Wl[i][j] for j in range(n)] for i in range(n)]
    levels = sorted(set(lab))
    k = len(levels)
    if k < 2:
        raise ValueError("need at least two colours")
    idx = {v: t for t, v in enumerate(levels)}
    ci = [idx[v] for v in lab]
    res = [[0.0] * k for _ in range(k)]
    for i in range(n):
        for j in range(n):
            if Wl[i][j] != 0.0:
                res[ci[i]][ci[j]] += Wl[i][j]
    res = [[v / 2.0 for v in r] for r in res]
    ntab = [float(ci.count(t)) for t in range(k)]
    N, s0, s1, s2 = _constants(Wl)
    n1, n2, n3 = N - 1.0, N - 2.0, N - 3.0
    sq = s0 * s0
    ejc = [s0 * a * (a - 1.0) / (2.0 * N * n1) for a in ntab]
    vjc = []
    for a, e in zip(ntab, ejc):
        v = s1 * a * (a - 1.0) / (N * n1)
        v += (s2 - 2.0 * s1) * a * (a - 1.0) * (a - 2.0) / (N * n1 * n2)
        v += (sq + s1 - s2) * a * (a - 1.0) * (a - 2.0) * (a - 3.0) / (N * n1 * n2 * n3)
        vjc.append(0.25 * v - e * e)
    ldiag = _pairs(list(range(k)), lambda i, j: res[i][j] + res[j][i])
    names = _pairs(levels, lambda a, b: f"{a}:{b}")
    prod = _pairs(ntab, lambda a, b: a * b)
    plus = _pairs(ntab, lambda a, b: a + b)
    prod2 = _pairs([a * (a - 1.0) for a in ntab], lambda a, b: a * b)
    ex = [s0 * p / (N * n1) for p in prod]
    var = []
    for p, q, r, e in zip(prod, plus, prod2, ex):
        v = 2.0 * s1 * p / (N * n1)
        v += (s2 - 2.0 * s1) * p * (q - 2.0) / (N * n1 * n2)
        v += 4.0 * (sq + s1 - s2) * r / (N * n1 * n2 * n3)
        var.append(0.25 * v - e * e)
    d3 = N * n1 * n2 * n3
    jvar = (s2 / (N * n1) - 4.0 * (sq + s1 - s2) * n1 / d3) * sum(prod)
    jvar += (
        4.0
        * ((s1 - s2) / d3 + 2.0 * sq * (2.0 * n - 3.0) / ((N * n1) * d3))
        * sum(_pairs([a * a for a in ntab], lambda a, b: a * b))
    )
    if k > 2:
        t3 = sum(ntab[a] * ntab[b] * ntab[c] for a, b, c in combinations(range(k), 3))
        jvar += (
            (2.0 * s1 - 5.0 * s2) / (N * n1 * n2) + 12.0 * (sq + s1 - s2) / d3 + 8.0 * sq / ((N * n1 * n2) * n1)
        ) * t3
    if k > 3:
        t4 = sum(ntab[a] * ntab[b] * ntab[c] * ntab[d] for a, b, c, d in combinations(range(k), 4))
        jvar -= 8.0 * ((s1 - s2) / d3 + 2.0 * sq * (2.0 * N - 3.0) / ((N * n1) * d3)) * t4
    jvar *= 0.25
    jc = [res[t][t] for t in range(k)] + ldiag + [sum(ldiag)]
    ee = ejc + ex + [sum(ex)]
    vv = vjc + var + [jvar]
    zz = [(a - b) / c**0.5 if c > 0.0 else float("nan") for a, b, c in zip(jc, ee, vv)]
    return RichResult(
        payload={
            "rows": [f"{v}:{v}" for v in levels] + names + ["Jtot"],
            "joincount": jc,
            "expected": ee,
            "variance": vv,
            "z": zz,
            "levels": levels,
        }
    )


def cheatsheet() -> str:
    return "join_count_multi(labels, W) -> k-colour join counts with moments (spdep::joincount.multi)."
