# morie.fn -- function file (rootcoder007/morie)
"""Lempel-Ziv complexity of a sequence."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lempel_ziv_complexity"]


def lempel_ziv_complexity(y):
    """Count the distinct phrases a sequence needs to build itself.

    This is an algorithmic rather than statistical notion of
    randomness: a sequence is complex when it cannot be assembled
    cheaply from pieces of its own past.  A periodic signal parses into
    a handful of phrases however long it runs; noise needs a new phrase
    almost every step.  Unlike entropy it needs no distribution and no
    stationarity, which is why it is used on short physiological
    records where neither is available.

    Formula: ``C(s)`` is the number of phrases in the LZ76 exhaustive
    parse; the normalised value is ``C(s) log_a(n) / n`` for an
    alphabet of size ``a``.

    Parameters
    ----------
    y : array-like
        Sequence; values are compared for equality, so a binarised
        series is the usual input.

    Returns
    -------
    RichResult
        ``estimate`` (phrase count), ``normalized``, ``alphabet``,
        ``n``.

    References
    ----------
    Lempel, A. & Ziv, J. (1976).  On the complexity of finite
    sequences.  IEEE Transactions on Information Theory 22:75-81.
    """
    s = [repr(v) for v in C.vec(y)]
    n = len(s)
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i = 0
                k = kmax = 1
            else:
                k = 1
    a = len(set(s))
    norm = c * (math.log(n) / math.log(a)) / n if a > 1 and n > 1 else float("nan")
    return RichResult(payload={
        "estimate": float(c), "normalized": norm, "alphabet": a, "n": n,
        "method": "Lempel-Ziv complexity, LZ76 parse"})


def cheatsheet():
    return "lzcomp: Lempel-Ziv complexity of a sequence."
