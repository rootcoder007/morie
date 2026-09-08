"""Joint entropy H(X,Y)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["joint_entropy"]


def joint_entropy(pxy, base=2.0):
    """
    Joint entropy of a 2-D joint pmf.

    Formula: H(X,Y) = -sum p(x,y) log p(x,y)

    Verified against Cover & Thomas (2006) eq. (2.15) p. 17 -- source
    consulted.

    Parameters
    ----------
    pxy : nested sequence
        Joint pmf ``p[i][j]``; normalised internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, hx, hy, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.15).
    """
    nx, ny = _big2.dims2(pxy)
    tot = float(sum(float(v) for r in pxy for v in r))
    if not (tot > 0.0):
        raise ValueError("pxy must have positive total mass")
    p = [[float(pxy[i][j]) / tot for j in range(ny)] for i in range(nx)]
    return RichResult(
        payload={
            "estimate": _big2.entropy([v for r in p for v in r], base),
            "hx": _big2.entropy(_big2.marg2(p, 0), base),
            "hy": _big2.entropy(_big2.marg2(p, 1), base),
            "n": nx * ny,
            "method": "Joint entropy H(X,Y) -- Cover & Thomas (2006) eq. (2.15)",
        }
    )


def cheatsheet():
    return "jntent: Joint entropy H(X,Y)"
