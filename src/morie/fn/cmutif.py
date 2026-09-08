"""Conditional mutual information I(X;Y|Z)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["conditional_mi"]



def conditional_mi(pxyz, base=2.0):
    """
    Conditional mutual information of a 3-D joint pmf.

    Formula: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Verified against Cover & Thomas (2006) eq. (2.60)-(2.61) p. 23 --
    source consulted. The book writes I(X;Y|Z) = H(X|Z) - H(X|Y,Z),
    which expands to the four-entropy form used here.

    Parameters
    ----------
    pxyz : nested sequence
        Joint pmf ``p[i][j][k]``; normalised internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, hxz, hyz, hxyz, hz, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.60).
    """
    nx, ny, nz = _big2.dims3(pxyz)
    tot = float(sum(_big2.flat3(pxyz)))
    if not (tot > 0.0):
        raise ValueError("pxyz must have positive total mass")
    p = [[[float(pxyz[i][j][k]) / tot for k in range(nz)] for j in range(ny)] for i in range(nx)]
    hxyz = _big2.entropy(_big2.flat3(p), base)
    hxz = _big2.entropy(_big2.marg3(p, (0, 2)), base)
    hyz = _big2.entropy(_big2.marg3(p, (1, 2)), base)
    hz = _big2.entropy(_big2.marg3(p, (2,)), base)
    return RichResult(
        payload={
            "estimate": hxz + hyz - hxyz - hz,
            "hxz": hxz,
            "hyz": hyz,
            "hxyz": hxyz,
            "hz": hz,
            "n": nx * ny * nz,
            "method": "Conditional mutual information I(X;Y|Z) -- Cover & Thomas (2006) eq. (2.60)",
        }
    )


def cheatsheet():
    return "cmutif: Conditional mutual information I(X;Y|Z)"


# compact alias per ledger/NAMING.md
conditionalmi = conditional_mi
