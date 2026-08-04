# morie.fn -- function file (rootcoder007/morie)
"""Aitchison inner product of two compositions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compip", "aitchison_inner_product"]


def compip(x, y):
    """Aitchison inner product on the simplex.

    This is the inner product that makes the simplex a Euclidean vector
    space under perturbation and powering; every distance, angle and
    projection in compositional analysis descends from it.

    Formula: <x, y>_a = sum_i clr(x)_i clr(y)_i
                      = (1/D) sum_{i<j} log(x_i/x_j) log(y_i/y_j)

    Parameters
    ----------
    x, y : array-like
        Strictly positive vectors of parts, the same length.

    Returns
    -------
    RichResult
        ``inner``, ``inner_pairwise``, ``cos_angle``, ``D``.  The two
        inner products agree; both are returned as a self-check.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``scalar`` applies the
    centred log-ratio to both arguments and sums the elementwise
    product.
    """
    x = C.vec(x)
    y = C.vec(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of parts")
    if any(v <= 0 for v in x) or any(v <= 0 for v in y):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    Lx = [math.log(v) for v in x]
    Ly = [math.log(v) for v in y]
    mx = sum(Lx) / D
    my = sum(Ly) / D
    zx = [v - mx for v in Lx]
    zy = [v - my for v in Ly]
    ip = sum(a * b for a, b in zip(zx, zy))
    pw = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            pw += (Lx[i] - Lx[j]) * (Ly[i] - Ly[j])
    nx = math.sqrt(sum(v * v for v in zx))
    ny = math.sqrt(sum(v * v for v in zy))
    cos = ip / (nx * ny) if nx > 0 and ny > 0 else float("nan")
    return RichResult(payload={
        "inner": ip, "inner_pairwise": pw / D, "cos_angle": cos, "D": D,
        "method": "Aitchison inner product"})


aitchison_inner_product = compip


def cheatsheet():
    return "aitip: <x,y>_a = sum clr(x) clr(y)"
