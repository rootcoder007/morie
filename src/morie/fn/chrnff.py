# morie.fn -- function file (rootcoder007/morie)
"""Chernoff bound on an upper tail."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['chernbnd', 'chernoff_bound']


def chernbnd(mgf, a, s_grid=None):
    """Chernoff bound on an upper tail.

    The bound is only as good as the optimisation over s, so the grid is an explicit argument rather than a hidden constant, and the minimising s is returned. A grid that bottoms out at its own endpoint has not found the optimum, and ``at_boundary`` says so instead of quietly returning a loose bound.


    Formula: P(X >= a) <= min_{s>0} exp(-s a) E[exp(s X)]

    Parameters
    ----------
    mgf : callable
        The moment generating function s -> E[exp(s X)].
    a : float
        Tail threshold.
    s_grid : array-like, optional
        Positive values of s searched; a fixed geometric grid over (0.01, 8.7] if omitted.

    Returns
    -------
    RichResult
        ``bound``, ``s``, ``log_bound``, ``at_boundary``.

    References
    ----------
    Chernoff (1952), A measure of asymptotic efficiency for tests of a
    hypothesis based on the sum of observations, Annals of Mathematical
    Statistics 23:493-507.  Not held locally; the exponential Markov
    bound is stated in this exact form in every standard reference.
    """
    a = float(a)
    if s_grid is None:
        grid = [0.01 * (1.05 ** k) for k in range(141)]
    else:
        grid = C.vec(s_grid)
    if any(v <= 0 for v in grid):
        raise ValueError("s must be positive")
    best, bs = float("inf"), float("nan")
    for s in grid:
        try:
            v = math.exp(-s * a) * float(mgf(s))
        except OverflowError:
            continue
        if v < best:
            best, bs = v, s
    if best == float("inf"):
        raise ValueError("mgf overflowed at every grid point")
    return RichResult(payload={
        "bound": best, "s": bs,
        "log_bound": math.log(best) if best > 0 else float("-inf"),
        "at_boundary": bs in (grid[0], grid[-1]),
        "method": "Chernoff bound"})


chernoff_bound = chernbnd


def cheatsheet():
    return "chrnff: Chernoff bound on an upper tail."
