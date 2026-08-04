"""Conditional entropy H(Y|X)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["conditional_entropy"]



def conditional_entropy(pxy, base=2.0):
    """
    Conditional entropy H(Y|X) of a 2-D joint pmf.

    Formula: H(Y|X) = H(X,Y) - H(X)

    Verified against Cover & Thomas (2006) eq. (2.10)/(2.12) p. 17 and
    the chain rule H(X,Y) = H(X) + H(Y|X), eq. (2.14) -- source consulted.

    Parameters
    ----------
    pxy : nested sequence
        Joint pmf ``p[i][j]``; normalised internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, hxy, hx, hy, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.10), (2.12).
    """
    nx, ny = _big2.dims2(pxy)
    tot = float(sum(float(v) for r in pxy for v in r))
    if not (tot > 0.0):
        raise ValueError("pxy must have positive total mass")
    p = [[float(pxy[i][j]) / tot for j in range(ny)] for i in range(nx)]
    hxy = _big2.entropy([v for r in p for v in r], base)
    hx = _big2.entropy(_big2.marg2(p, 0), base)
    hy = _big2.entropy(_big2.marg2(p, 1), base)
    return RichResult(
        payload={
            "estimate": hxy - hx,
            "hxy": hxy,
            "hx": hx,
            "hy": hy,
            "n": nx * ny,
            "method": "Conditional entropy H(Y|X) -- Cover & Thomas (2006) eq. (2.12)",
        }
    )


def cheatsheet():
    return "cndent: Conditional entropy H(Y|X)"
