# morie.fn -- function file (rootcoder007/morie)
"""Wilcoxon rank-sum (Mann-Whitney) test."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ranksum", "wasserman_wilcoxon"]


def ranksum(x, y, correct=True):
    """Wilcoxon rank-sum test, normal approximation with a tie correction.

    The tie correction is not cosmetic: with heavy ties the uncorrected
    variance is too large, the statistic too small, and the test
    conservative in a way that quietly costs power.  Ties are given
    average ranks, which is what makes the correction the right one.

    This tests a shift in DISTRIBUTION, not in mean -- with unequal
    spreads it can reject when the means are identical.

    Formula: W = sum of the ranks of x in the pooled sample;
             E[W] = n1(n1 + n2 + 1)/2;
             Var[W] = n1 n2 (N + 1)/12
                      - n1 n2 sum(t^3 - t) / (12 N (N - 1));
             z = (W - E[W] -+ 1/2) / sd, continuity-corrected

    Parameters
    ----------
    x, y : array-like
        The two samples.
    correct : bool
        Apply the 1/2 continuity correction.

    Returns
    -------
    RichResult
        ``statistic`` (W), ``U``, ``z``, ``p_value``, ``expected``,
        ``variance``, ``n1``, ``n2``, ``n_tied_groups``.

    References
    ----------
    Wilcoxon (1945), Individual comparisons by ranking methods,
    Biometrics Bulletin 1(6), 80-83, and Mann & Whitney (1947), On a
    test of whether one of two random variables is stochastically
    larger than the other, Annals of Mathematical Statistics 18(1),
    50-60 -- the primary sources.  Wasserman (2004), All of Statistics,
    does NOT contain the rank-sum test; the full text of the book was
    fetched and searched to establish that, so it is not cited here.
    """
    x = C.vec(x)
    y = C.vec(y)
    n1 = len(x)
    n2 = len(y)
    if n1 < 1 or n2 < 1:
        raise ValueError("both samples must be non-empty")
    pool = x + y
    N = n1 + n2
    order = sorted(range(N), key=lambda i: pool[i])
    rank = [0.0] * N
    i = 0
    tiesum = 0.0
    groups = 0
    while i < N:
        j = i
        while j < N and pool[order[j]] == pool[order[i]]:
            j += 1
        r = (i + j + 1) / 2.0
        for t in range(i, j):
            rank[order[t]] = r
        tcount = j - i
        if tcount > 1:
            groups += 1
            tiesum += tcount ** 3 - tcount
        i = j
    W = sum(rank[:n1])
    E = n1 * (N + 1) / 2.0
    V = n1 * n2 * (N + 1) / 12.0 - n1 * n2 * tiesum / (12.0 * N * (N - 1))
    if V <= 0:
        raise ValueError("the rank variance is zero; every value is tied")
    d = W - E
    cc = 0.5 if correct else 0.0
    if d > 0:
        z = (d - cc) / math.sqrt(V)
    elif d < 0:
        z = (d + cc) / math.sqrt(V)
    else:
        z = 0.0
    return RichResult(payload={
        "statistic": W, "U": W - n1 * (n1 + 1) / 2.0, "z": z,
        "p_value": 2.0 * (1.0 - C.pnorm(abs(z))), "expected": E,
        "variance": V, "n1": float(n1), "n2": float(n2),
        "n_tied_groups": float(groups),
        "method": "Wilcoxon rank-sum, normal approximation with tie correction"})


wasserman_wilcoxon = ranksum


def cheatsheet():
    return "wsmwlx: W = rank sum of x; V corrected by sum(t^3-t)/(12N(N-1))"
