# morie.fn -- tail3 batch (rootcoder007/morie)
"""Local-shift sensitivity of an estimator.

Source consulted: Hampel, F.R., Ronchetti, E.M., Rousseeuw, P.J. & Stahel,
W.A. (1986). *Robust Statistics: The Approach Based on Influence Functions*.
Wiley, section 2.1c.  The local-shift sensitivity of ``T`` at ``F`` is

    lambda* = sup_{x != y} |IF(y; T, F) - IF(x; T, F)| / |y - x|

the worst effect of shifting an observation slightly (rounding, grouping,
local wiggling of the data).
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["local_shift"]


def local_shift(IF, x=None):
    """Local-shift sensitivity lambda*.

    Parameters
    ----------
    IF : array-like
        Influence-function values on a grid.
    x : array-like, optional
        Grid points.  Defaults to ``0, 1, ..., n-1``.

    Returns
    -------
    RichResult
        estimate (lambda*), lambda_star, i, j, n, method.

    References
    ----------
    Hampel, Ronchetti, Rousseeuw & Stahel (1986), section 2.1c.
    """
    IF = np.atleast_1d(np.asarray(IF, dtype=float)).ravel()
    n = int(IF.size)
    if x is None:
        xg = np.asarray([float(i) for i in range(n)], dtype=float)
    else:
        xg = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    best = float("-inf")
    bi = 0
    bj = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(xg[j]) - float(xg[i])
            if dx == 0.0:
                continue
            v = abs(float(IF[j]) - float(IF[i])) / abs(dx)
            if v > best:
                best = v
                bi = i
                bj = j
    lam = float(best) if n > 1 else float("nan")
    return RichResult(
        payload={
            "estimate": lam,
            "lambda_star": lam,
            "i": bi,
            "j": bj,
            "n": n,
            "method": "Local-shift sensitivity (Hampel et al. 1986)",
        }
    )


# CANONICAL TEST
# >>> r = local_shift([0.0, 1.0, 1.5], x=[0.0, 1.0, 2.0])
# >>> assert abs(r["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "localS(IF, x): local-shift sensitivity lambda*."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
localshift = local_shift
