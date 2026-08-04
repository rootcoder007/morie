# morie.fn -- function file (rootcoder007/morie)
"""One-dimensional Wasserstein-1 distance."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ot_wasserstein_1d"]


def ot_wasserstein_1d(x, y):
    """Optimal transport cost on the line, which needs no solver.

    In one dimension the optimal coupling is always the monotone one:
    sort both samples and match them in order.  No linear program is
    needed, which is why the one-dimensional case is used as a building
    block (sliced Wasserstein) for problems that would otherwise be
    intractable.

    Formula: ``W_1 = integral |F(t) - G(t)| dt``, which for equal sample
    sizes equals ``mean_i |x_(i) - y_(i)|``.

    Parameters
    ----------
    x, y : array-like
        Samples; equal length.

    Returns
    -------
    RichResult
        ``W1``, ``estimate`` (same value), ``n``.

    References
    ----------
    Vallender, S. S. (1973).  Calculation of the Wasserstein distance
    between probability distributions on the line.  Theory of
    Probability and its Applications 18:784-786.
    """
    xs = sorted(C.vec(x))
    ys = sorted(C.vec(y))
    n = len(xs)
    w = sum(abs(xs[i] - ys[i]) for i in range(n)) / n
    return RichResult(payload={"W1": w, "estimate": w, "n": n,
                               "method": "One-dimensional Wasserstein-1 distance"})


def cheatsheet():
    return "otws: One-dimensional Wasserstein-1 distance."
