"""Differential entropy h(X)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["differential_entropy"]



def differential_entropy(density, x=None, base=2.0):
    """
    Differential entropy of a density given on a grid.

    Formula: h(X) = -integral f(x) log f(x) dx

    Verified against Cover & Thomas (2006), *Elements of Information
    Theory*, 2nd ed., eq. (8.1) p. 243 -- source consulted.

    The integral is a trapezoid rule on the supplied grid (a fixed node
    set, never adapted); the convention 0 log 0 = 0 is used.

    Parameters
    ----------
    density : array-like
        Non-negative density values f(x) at the grid points.
    x : array-like, optional
        Grid; defaults to ``linspace(0, 1, len(density))``.
    base : float, optional
        Log base; 2 gives bits, ``None`` gives nats.

    Returns
    -------
    RichResult
        Keys: estimate, mass, n, base, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (8.1).
    """
    f = np.atleast_1d(np.asarray(density, dtype=float))
    n = len(f)
    if n < 2:
        raise ValueError("density needs at least two grid points")
    grid = np.linspace(0.0, 1.0, n) if x is None else np.atleast_1d(np.asarray(x, dtype=float))
    if len(grid) != n:
        raise ValueError("x and density must have the same length")
    if float(np.min(f)) < 0.0:
        raise ValueError("density must be non-negative")
    h = float(np.trapezoid(-_big2.xlogx(f, base), grid))
    mass = float(np.trapezoid(f, grid))
    return RichResult(
        payload={
            "estimate": h,
            "mass": mass,
            "n": n,
            "base": (None if base is None else float(base)),
            "method": "Differential entropy h(X) -- Cover & Thomas (2006) eq. (8.1)",
        }
    )


def cheatsheet():
    return "acvent: Differential entropy h(X)"
