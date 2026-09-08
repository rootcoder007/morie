# morie.fn -- function file (rootcoder007/morie)
"""Powering: scalar multiplication on the simplex."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["powering", "aitchison_powering"]


def powering(a, x, total=1.0):
    """Power a composition by a real scalar.

    Powering is scalar multiplication in the Aitchison vector space:
    together with perturbation it makes the simplex a real vector
    space, so "twice the effect" has an unambiguous meaning.  a > 1
    pushes a composition towards its dominant part, 0 < a < 1 pulls it
    towards the barycentre, and a < 0 reverses it.

    Formula: a (.) x = C( x_1^a, ..., x_D^a )

    Parameters
    ----------
    a : float
        Scalar.
    x : array-like
        Strictly positive vector of parts.
    total : float
        Constant the result sums to.

    Returns
    -------
    RichResult
        ``composition``, ``a``, ``total``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 2.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose power operator on an
    ``acomp`` closes the elementwise power.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    a = float(a)
    p = [v ** a for v in x]
    s = sum(p)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in p], "a": a, "total": k,
        "D": len(x), "method": "Powering on the simplex"})


aitchison_powering = powering


def cheatsheet():
    return "aitpow: a (.) x = C(x^a)"
