# morie.fn -- function file (rootcoder007/morie)
"""Jaro-Winkler string similarity."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["jaro_winkler"]


def jaro_winkler(s1, s2, p=0.1, max_prefix=4):
    """String similarity that rewards a shared prefix.

    Winkler adjustment exists because of one empirical fact about the
    data it was built for: people mistype the ends of names far more
    often than the beginnings.  So a shared prefix is evidence, and the
    boost is capped at four characters to keep the score bounded and to
    stop long common prefixes from swamping the rest of the comparison.

    Formula: with ``m`` matches within
    ``floor(max(|s1|, |s2|) / 2) - 1`` positions and ``t`` half the
    transpositions,
    ``jaro = (m/|s1| + m/|s2| + (m - t)/m) / 3`` and
    ``jw = jaro + l p (1 - jaro)`` for prefix length ``l``.

    Parameters
    ----------
    s1, s2 : str
        Strings to compare.
    p : float, default 0.1
        Prefix scaling factor.
    max_prefix : int, default 4
        Longest prefix that earns a boost.

    Returns
    -------
    RichResult
        ``estimate`` (Jaro-Winkler), ``jaro``, ``matches``,
        ``transpositions``, ``prefix``.

    References
    ----------
    Winkler, W. E. (1990).  String comparator metrics and enhanced
    decision rules in the Fellegi-Sunter model of record linkage.
    Proceedings of the Section on Survey Research Methods, American
    Statistical Association, 354-359.  The base measure is Jaro, M. A.
    (1989), JASA 84:414-420.
    """
    a, b = str(s1), str(s2)
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return RichResult(payload={"estimate": 0.0, "jaro": 0.0, "matches": 0,
                                   "transpositions": 0, "prefix": 0,
                                   "method": "Jaro-Winkler string similarity"})
    win = max(la, lb) // 2 - 1
    if win < 0:
        win = 0
    fa = [False] * la
    fb = [False] * lb
    m = 0
    for i in range(la):
        lo = max(0, i - win)
        hi = min(lb - 1, i + win)
        for j in range(lo, hi + 1):
            if not fb[j] and a[i] == b[j]:
                fa[i] = fb[j] = True
                m += 1
                break
    if m == 0:
        return RichResult(payload={"estimate": 0.0, "jaro": 0.0, "matches": 0,
                                   "transpositions": 0, "prefix": 0,
                                   "method": "Jaro-Winkler string similarity"})
    ka = [a[i] for i in range(la) if fa[i]]
    kb = [b[j] for j in range(lb) if fb[j]]
    t = sum(1 for i in range(m) if ka[i] != kb[i]) / 2.0
    jaro = (m / la + m / lb + (m - t) / m) / 3.0
    l = 0
    for i in range(min(max_prefix, la, lb)):
        if a[i] == b[i]:
            l += 1
        else:
            break
    return RichResult(payload={
        "estimate": jaro + l * p * (1.0 - jaro), "jaro": jaro, "matches": m,
        "transpositions": t, "prefix": l,
        "method": "Jaro-Winkler string similarity"})


jarowinkler = jaro_winkler


def cheatsheet():
    return "jarow: Jaro-Winkler string similarity."
