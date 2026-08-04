"""Mutual information I(X;Y)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["mutual_information"]


def mutual_information(pxy, base=2.0):
    """
    Mutual information of a 2-D joint pmf.

    Formula: I(X;Y) = sum p(x,y) log[p(x,y)/(p(x)p(y))]

    Verified against Shannon (1948) Section 12 (the rate
    R = H(x) - H_y(x)) and Cover & Thomas (2006) eq. (2.28)-(2.30)
    p. 20 -- sources consulted.

    Parameters
    ----------
    pxy : nested sequence
        Joint pmf ``p[i][j]``; normalised internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, hx, hy, hxy, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.28)-(2.30).
    """
    nx, ny = _big2.dims2(pxy)
    tot = float(sum(float(v) for r in pxy for v in r))
    if not (tot > 0.0):
        raise ValueError("pxy must have positive total mass")
    p = [[float(pxy[i][j]) / tot for j in range(ny)] for i in range(nx)]
    hx = _big2.entropy(_big2.marg2(p, 0), base)
    hy = _big2.entropy(_big2.marg2(p, 1), base)
    hxy = _big2.entropy([v for r in p for v in r], base)
    return RichResult(
        payload={
            "estimate": hx + hy - hxy,
            "hx": hx,
            "hy": hy,
            "hxy": hxy,
            "n": nx * ny,
            "method": "Mutual information I(X;Y) -- Cover & Thomas (2006) eq. (2.28)",
        }
    )


def cheatsheet():
    return "mutinf: Mutual information I(X;Y)"
