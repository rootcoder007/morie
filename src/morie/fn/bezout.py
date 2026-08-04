# morie.fn -- function file (rootcoder007/morie)
"""Bezout coefficients by the extended Euclidean algorithm."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bezout']


def bezout(a, b):
    """Bezout coefficients by the extended Euclidean algorithm.

    The identity is returned, not asserted: x and y come back next to the gcd so a caller can multiply out and check. Integer arithmetic throughout, so the result is exact for inputs of any size rather than accurate to a few significant figures.


    Formula: a x + b y = gcd(a, b)

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    RichResult
        ``gcd``, ``x``, ``y``, ``check`` (a x + b y), ``a``, ``b``.

    References
    ----------
    Bezout (1779), Theorie generale des equations algebriques.  Not
    held locally; the extended Euclidean algorithm and the identity
    a x + b y = gcd(a, b) are standard published results.
    """
    a0 = int(a); b0 = int(b)
    old_r, r = a0, b0
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return RichResult(payload={
        "gcd": old_r, "x": old_s, "y": old_t,
        "check": a0 * old_s + b0 * old_t, "a": a0, "b": b0,
        "method": "Bezout coefficients (extended Euclid)"})



def cheatsheet():
    return "bezout: Bezout coefficients by the extended Euclidean algorithm."
