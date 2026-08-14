# SPDX-License-Identifier: AGPL-3.0-or-later
"""HOT SAX time-series discord discovery."""

import math

from . import _array_core as np

from ._richresult import RichResult
from .saxR import saxR

__all__ = ["hot", "hot_sax"]


def _znorm(seg):
    m = sum(seg) / len(seg)
    v = sum((s - m) ** 2 for s in seg) / len(seg)
    sd = math.sqrt(v)
    if sd < 1e-12:
        return [0.0] * len(seg)
    return [(s - m) / sd for s in seg]


def _dist(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def hot(x, window, alphabet=3, word_length=None):
    """
    Time-series discord discovery (HOT SAX).

    The discord (Keogh, Lin & Fu 2005, Definition 6) is the
    subsequence D of length n whose distance to its nearest non-self
    match (Definition 5: |p - q| >= n) is largest:
    argmax_D min over non-self matches C of Dist(D, C), with
    z-normalised Euclidean distance (Definition 8 and Sec. 2.1: each
    subsequence is normalised to mean 0, sd 1 before the distance).

    Search: the outer/inner heuristic of their Table 2 -- outer loop
    visits subsequences whose SAX word is rarest first (Sec. 4.2),
    inner loop visits same-word subsequences first (Sec. 4.3), with
    early abandoning (Observations 1-4). The heuristic changes only
    the running time; the result is guaranteed identical to the brute
    force of their Table 1 (Sec. 3), which is how the tests anchor it.
    Remaining candidates are visited in index order (deterministic
    stand-in for the arbitrary order allowed by the paper).

    Parameters
    ----------
    x : array-like
        Series.
    window : int
        Discord length n (the single intuitive parameter).
    alphabet : int
        SAX alphabet size a (default 3, their Sec. 4.4 finding).
    word_length : int, optional
        SAX word length w; default min(3, window) frames chosen so
        the window is divisible (efficiency only, never affects the
        result).

    Returns
    -------
    result : RichResult
        Keys: location (1-based start of the discord), distance
        (to nearest non-self match), neighbor (1-based start of that
        nearest non-self match).

    References
    ----------
    Keogh, E., Lin, J. and Fu, A. (2005), "HOT SAX: efficiently
    finding the most unusual time series subsequence", Proc. 5th IEEE
    International Conference on Data Mining (ICDM 2005), pp. 226-233.
    Definitions 5-8, Tables 1-2, Sections 3-4 (long version).
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    keogh-lin-fu-2005-hotsax-discords-icdm-longver.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    xs = [float(v) for v in xv]
    m = len(xs)
    n = int(window)
    if n < 2 or n > m // 2:
        raise ValueError("need 2 <= window <= len(x)/2")
    nsub = m - n + 1
    subs = [_znorm(xs[p:p + n]) for p in range(nsub)]
    # SAX words for the ordering heuristic
    if word_length is None:
        word_length = 1
        for w in (3, 2):
            if n % w == 0:
                word_length = w
                break
    words = []
    for p in range(nsub):
        words.append(saxR(xs[p:p + n], word_length, alphabet)["word"])
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    # outer: rarest-word subsequences first, then index order
    outer = sorted(range(nsub), key=lambda p: (counts[words[p]], p))
    by_word = {}
    for p in range(nsub):
        by_word.setdefault(words[p], []).append(p)
    best_dist = -1.0
    best_loc = -1
    best_nb = -1
    for p in outer:
        nnd = math.inf
        nnq = -1
        # inner: same-word first, then index order
        inner = by_word[words[p]] + [q for q in range(nsub)
                                     if words[q] != words[p]]
        for q in inner:
            if abs(p - q) < n:
                continue
            d = _dist(subs[p], subs[q])
            if d < best_dist:
                # Observation 1/line 9 of Table 2: p cannot be the
                # discord; abandon this inner loop.
                nnd = -math.inf
                break
            if d < nnd:
                nnd = d
                nnq = q
        if nnd > best_dist and nnq >= 0:
            best_dist = nnd
            best_loc = p
            best_nb = nnq
    return RichResult(payload={
        "location": best_loc + 1,
        "distance": float(best_dist),
        "neighbor": best_nb + 1,
        "window": n,
        "estimate": best_loc + 1,
        "n": m,
        "method": "HOT SAX discord (Keogh-Lin-Fu 2005)",
    })


def hot_sax(x, window, alphabet=3, word_length=None):
    """Alias for hot (original stub export name)."""
    return hot(x, window, alphabet=alphabet, word_length=word_length)


def cheatsheet():
    return "hot(x, window) -> most unusual subsequence (discord) via HOT SAX search"

# public names resolved by fn/_lazy_map.json
hotsax = hot
