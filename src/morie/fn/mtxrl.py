# morie.fn -- function file (rootcoder007/morie)
"""Value of a two-person zero-sum matrix game."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["matgame", "matrix_game"]


def matgame(A, iters=2000):
    """Value and optimal strategies of a zero-sum matrix game.

    The minimax theorem guarantees a value exists in MIXED strategies;
    in pure strategies it usually does not, and the gap between
    ``maximin`` and ``minimax`` is exactly the evidence of that.  When
    the two coincide the game has a saddle point and the answer is
    exact -- that case is detected and returned without iterating.

    Otherwise fictitious play is run for a FIXED number of rounds, and
    what is returned with it is a rigorous BRACKET, not just an
    estimate: for the empirical strategies (x, y) the value must lie
    between min_j (x'A)_j and max_i (Ay)_i.  A caller can therefore see
    how far from converged the answer is instead of taking it on
    trust.

    Formula: v = max_x min_y x' A y = min_y max_x x' A y;
             fictitious play: each side best-responds to the opponent's
             empirical frequencies, ties to the lowest index

    Parameters
    ----------
    A : array-like, shape (m, n)
        Payoff matrix to the ROW player; the column player pays it.
    iters : int
        Fixed number of fictitious-play rounds.

    Returns
    -------
    RichResult
        ``value``, ``lower``, ``upper``, ``row_strategy``,
        ``col_strategy``, ``maximin``, ``minimax``, ``saddle`` (1 when
        a pure saddle point exists), ``iterations``, ``m``, ``n``.

    References
    ----------
    von Neumann, J. (1928), Zur Theorie der Gesellschaftsspiele,
    Mathematische Annalen 100, 295-320 -- the minimax theorem, that
    every finite two-person zero-sum game has a value in mixed
    strategies.  The fictitious-play iteration used when no saddle
    point exists is Brown (1951), Iterative solution of games by
    fictitious play, in Activity Analysis of Production and
    Allocation, and its convergence for zero-sum games is Robinson
    (1951), An iterative method of solving a game, Annals of
    Mathematics 54(2), 296-301; neither is von Neumann's and both are
    cited to their own sources.
    """
    A = C.mat(A)
    m = len(A)
    if m < 1:
        raise ValueError("the payoff matrix must be non-empty")
    n = len(A[0])
    if any(len(r) != n for r in A):
        raise ValueError("the payoff matrix must be rectangular")
    rowmin = [min(r) for r in A]
    colmax = [max(A[i][j] for i in range(m)) for j in range(n)]
    maximin = max(rowmin)
    minimax = min(colmax)
    if abs(maximin - minimax) < 1e-15:
        i0 = max(range(m), key=lambda i: (rowmin[i], -i))
        j0 = min(range(n), key=lambda j: (colmax[j], j))
        x = [1.0 if i == i0 else 0.0 for i in range(m)]
        y = [1.0 if j == j0 else 0.0 for j in range(n)]
        return RichResult(payload={
            "value": maximin, "lower": maximin, "upper": minimax,
            "row_strategy": x, "col_strategy": y, "maximin": maximin,
            "minimax": minimax, "saddle": 1.0, "iterations": 0.0,
            "m": float(m), "n": float(n),
            "method": "Matrix game with a pure saddle point"})
    cr = [0] * m
    cc = [0] * n
    urow = [0.0] * n
    ucol = [0.0] * m
    i = 0
    cr[i] = 1
    for j in range(n):
        urow[j] += A[i][j]
    T = int(iters)
    for _ in range(T):
        j = min(range(n), key=lambda t: (urow[t], t))
        cc[j] += 1
        for t in range(m):
            ucol[t] += A[t][j]
        i = max(range(m), key=lambda t: (ucol[t], -t))
        cr[i] += 1
        for t in range(n):
            urow[t] += A[i][t]
    sr = sum(cr)
    sc = sum(cc)
    x = [v / sr for v in cr]
    y = [v / sc for v in cc]
    Ay = [sum(A[t][j] * y[j] for j in range(n)) for t in range(m)]
    xA = [sum(x[t] * A[t][j] for t in range(m)) for j in range(n)]
    lo = min(xA)
    hi = max(Ay)
    return RichResult(payload={
        "value": 0.5 * (lo + hi), "lower": lo, "upper": hi,
        "row_strategy": x, "col_strategy": y, "maximin": maximin,
        "minimax": minimax, "saddle": 0.0, "iterations": float(T),
        "m": float(m), "n": float(n),
        "method": "Matrix game by fictitious play with a rigorous bracket"})


matrix_game = matgame


def cheatsheet():
    return "mtxrl: v = max_x min_y x'Ay; saddle exact, else bracketed fictitious play"
