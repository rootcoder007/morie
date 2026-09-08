# SPDX-License-Identifier: AGPL-3.0-or-later
"""SAX -- Symbolic Aggregate approXimation."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._rng import normal_quantile

__all__ = ["saxR", "sax_representation", "sax_breakpoints", "sax_mindist"]

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sax_breakpoints(alphabet):
    """
    Breakpoints B = beta_1 .. beta_{a-1}: sorted numbers such that the
    area under the N(0,1) curve from beta_i to beta_{i+1} is 1/a
    (Lin et al. 2003, Definition 1; their Table 3 lists the values
    rounded to 2 decimals). Computed exactly as Phi^{-1}(i/a) with the
    native AS 241 quantile.
    """
    a = int(alphabet)
    if a < 2:
        raise ValueError("alphabet size must be >= 2")
    return [float(normal_quantile(i / a)) for i in range(1, a)]


def _paa(z, w):
    # eq (1): cbar_i = (w/n) sum_{j = n/w (i-1)+1}^{n/w i} c_j
    n = len(z)
    if n % w != 0:
        raise ValueError("word length must divide the series length "
                         "(eq 1 of Lin et al. 2003 uses equal frames)")
    f = n // w
    return [sum(z[i * f:(i + 1) * f]) / f for i in range(w)]


def saxR(x, window, alphabet, eps=1e-8):
    """
    SAX symbolic representation of a series.

    Steps (Lin, Keogh, Lonardi & Chiu 2003; journal version Lin,
    Keogh, Wei & Lonardi 2007):
    1. z-normalise the series to mean 0, sd 1 (their Sec. 3.1; if the
       standard deviation is below eps the whole word is assigned the
       middle-ranged symbol, their Sec. 3.4 special case).
    2. PAA, eq (1): mean of each of w equal frames.
    3. Discretise with breakpoints beta_i = Phi^{-1}(i/a)
       (Definition 1, Table 3): symbol j iff
       beta_{j-1} <= cbar_i < beta_j (Definition 2, eq 2).

    Parameters
    ----------
    x : array-like
        Series (length divisible by `window`).
    window : int
        Word length w (number of PAA frames).
    alphabet : int
        Alphabet size a > 2 recommended in 5..8 (their Sec. 3.5).
    eps : float
        Near-constant guard threshold.

    Returns
    -------
    result : RichResult
        Keys: word (string), symbols (0-based indices), paa,
        breakpoints, mean, sd.

    References
    ----------
    Lin, J., Keogh, E., Lonardi, S. and Chiu, B. (2003), "A symbolic
    representation of time series, with implications for streaming
    algorithms", Proc. 8th ACM SIGMOD Workshop on Research Issues in
    Data Mining and Knowledge Discovery (DMKD 2003), pp. 2-11.
    Equations (1)-(2), Definitions 1-2, Table 3. Journal version:
    Lin, J., Keogh, E., Wei, L. and Lonardi, S. (2007), "Experiencing
    SAX: a novel symbolic representation of time series", Data Mining
    and Knowledge Discovery 15(2), 107-144.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    lin-keogh-lonardi-chiu-2003-sax-dmkd.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(xv)
    w = int(window)
    a = int(alphabet)
    if w < 1 or w > n:
        raise ValueError("need 1 <= window <= n")
    bps = sax_breakpoints(a)
    mu = float(np.mean(xv))
    sd = math.sqrt(float(np.mean((xv - mu) ** 2)))
    if sd < eps:
        # near-constant subsequence: middle-ranged symbol throughout
        mid = (a - 1) // 2
        syms = [mid] * w
        paa = [0.0] * w
    else:
        z = [(float(v) - mu) / sd for v in xv]
        paa = _paa(z, w)
        syms = []
        for c in paa:
            j = 0
            for b in bps:
                if c >= b:
                    j += 1
            syms.append(j)
    word = "".join(_LETTERS[s] for s in syms)
    return RichResult(payload={
        "word": word,
        "symbols": syms,
        "paa": paa,
        "breakpoints": bps,
        "mean": mu,
        "sd": sd,
        "estimate": word,
        "n": n,
        "method": "SAX (Lin-Keogh-Lonardi-Chiu 2003)",
    })


def sax_mindist(word1, word2, n, alphabet):
    """
    MINDIST between two SAX words, eq (5):
    sqrt(n/w) * sqrt(sum_i dist(q_i, c_i)^2), with the cell lookup of
    eq (6): 0 if |r - c| <= 1, else beta_{max(r,c)-1} - beta_{min(r,c)}
    (1-based symbol indices).
    """
    if len(word1) != len(word2):
        raise ValueError("words must have equal length")
    w = len(word1)
    a = int(alphabet)
    bps = sax_breakpoints(a)
    tot = 0.0
    for s1, s2 in zip(word1, word2):
        r = _LETTERS.index(s1) + 1
        c = _LETTERS.index(s2) + 1
        if max(r, c) > a:
            raise ValueError("symbol outside alphabet")
        if abs(r - c) <= 1:
            d = 0.0
        else:
            d = bps[max(r, c) - 2] - bps[min(r, c) - 1]
        tot += d * d
    return math.sqrt(float(n) / w) * math.sqrt(tot)


def sax_representation(x, window, alphabet, eps=1e-8):
    """Alias for saxR (original stub export name)."""
    return saxR(x, window, alphabet, eps=eps)


def cheatsheet():
    return "saxR(x, w, a) -> SAX word via z-norm + PAA + Gaussian breakpoints; sax_mindist for word distance"
