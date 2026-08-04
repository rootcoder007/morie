# morie.fn -- function file (rootcoder007/morie)
"""Perturbation: the group operation on the simplex."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["perturb", "aitchison_perturbation"]


def perturb(x, y, total=1.0):
    """Perturbation of one composition by another.

    Perturbation is to the simplex what addition is to R^D: it is the
    group operation the whole Aitchison geometry is built on, and it is
    what a change of units, a dilution, or a compositional "effect"
    actually does to the data.

    Formula: x (+) y = C( x_1 y_1, ..., x_D y_D )

    Parameters
    ----------
    x, y : array-like
        Strictly positive vectors of parts, the same length.
    total : float
        Constant the result sums to.

    Returns
    -------
    RichResult
        ``composition``, ``total``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 2.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``perturbe`` is
    ``acomp(x * y)`` -- the elementwise product, then closed.
    """
    x = C.vec(x)
    y = C.vec(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of parts")
    if any(v <= 0 for v in x) or any(v <= 0 for v in y):
        raise ValueError("compositions must be strictly positive")
    p = [a * b for a, b in zip(x, y)]
    s = sum(p)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in p], "total": k, "D": len(x),
        "method": "Perturbation on the simplex"})


aitchison_perturbation = perturb


def cheatsheet():
    return "aitprt: x (+) y = C(x*y)"
