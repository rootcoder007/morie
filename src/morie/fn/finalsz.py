# morie.fn -- function file (rootcoder007/morie)
"""Final epidemic size (Kermack-McKendrick)."""

import math

from ._richresult import RichResult

__all__ = ["final_epidemic_size"]


def final_epidemic_size(R0, s0=1.0, i0=None, tol=1e-14, max_iter=200):
    """
    Final epidemic size (Kermack-McKendrick)

    Formula: for the standard SIR model the final size Z = S(0) - S(inf)
    satisfies the implicit relation

        Z = S(0) (1 - exp(-R0 [Z + I(0)]))

    (Ma & Earn 2006, eq. 4, p.681), which in the limit I(0) -> 0,
    S(0) -> 1 collapses to the classical

        Z = 1 - exp(-R0 Z)

    (their eq. 5).  Ma & Earn show the same relation holds for arbitrary
    distributions of the infectious period, so R0 alone determines the
    final size.  The root is isolated by bisection on [0, S(0)], where
    the residual is non-negative at 0 and strictly negative at S(0).

    Parameters
    ----------
    R0 : float
        Basic reproduction number (>= 0).
    s0 : float
        Initial susceptible proportion in (0, 1].
    i0 : float, optional
        Initial infectious proportion.  Default 1 - s0.
    tol : float
        Bracket width at which bisection stops.
    max_iter : int
        Maximum bisection steps.

    Returns
    -------
    result : dict
        Keys: estimate (final size Z), final_size, s_inf, attack_rate,
        R0, s0, i0, residual, iters, n, method.

    References
    ----------
    Kermack & McKendrick (1927), Proc. R. Soc. Lond. A 115(772):700-721,
    doi:10.1098/rspa.1927.0118.
    Ma & Earn (2006), Bull. Math. Biol. 68(3):679-702,
    doi:10.1007/s11538-005-9047-7.
    """
    R0 = float(R0)
    s0 = float(s0)
    if R0 < 0.0:
        raise ValueError("R0 must be non-negative")
    if not (0.0 < s0 <= 1.0):
        raise ValueError("s0 must lie in (0, 1]")
    i0 = (1.0 - s0) if i0 is None else float(i0)
    if i0 < 0.0 or s0 + i0 > 1.0:
        raise ValueError("i0 must be non-negative with s0 + i0 <= 1")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    def resid(Z):
        return s0 * (1.0 - math.exp(-R0 * (Z + i0))) - Z

    lo, hi = 0.0, s0
    it = 0
    if i0 == 0.0 and R0 * s0 <= 1.0:
        # no seed and R0 s0 <= 1: Z = 0 is the ONLY root in [0, s0]
        # (resid'(0) = R0 s0 - 1 <= 0 and resid is concave)
        Z = 0.0
    else:
        for it in range(1, int(max_iter) + 1):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        Z = 0.5 * (lo + hi)
    return RichResult(payload={
        "estimate": Z,
        "final_size": Z,
        "s_inf": s0 - Z,
        "attack_rate": Z / s0,
        "R0": R0,
        "s0": s0,
        "i0": i0,
        "residual": resid(Z),
        "iters": it,
        "n": 1,
        "method": "Final epidemic size (Kermack-McKendrick)",
    })


def cheatsheet():
    return "finalsz: Final epidemic size (Kermack-McKendrick)"


# compact alias per ledger/NAMING.md
finalepidemicsize = final_epidemic_size
