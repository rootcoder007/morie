"""Data processing inequality test."""

from . import _array_core as np
from . import _big2 as _big2
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["data_processing_inequality"]


def data_processing_inequality(pxyz, cdf=None, base=2.0):
    """
    Data-processing inequality for a Markov chain X -> Y -> Z.

    Formula: X->Y->Z implies I(X;Y) >= I(X;Z)

    Verified against Cover & Thomas (2006) Section 2.8, Theorem 2.8.1
    p. 34 -- source consulted. This is a deterministic identity, not a
    hypothesis test: it returns the two mutual informations, their gap,
    and I(X;Z|Y), which is zero exactly when the joint is Markov.

    Parameters
    ----------
    pxyz : nested sequence
        Joint pmf ``p[i][j][k]``; normalised internally.
    cdf : callable, optional
        Accepted and ignored, so older call sites keep working. The
        data-processing inequality carries no sampling distribution.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate (the gap I(X;Y)-I(X;Z)), ixy, ixz, markov_gap,
        holds, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Theorem 2.8.1.
    """
    nx, ny, nz = _big2.dims3(pxyz)
    tot = float(sum(_big2.flat3(pxyz)))
    if not (tot > 0.0):
        raise ValueError("pxyz must have positive total mass")
    p = [[[float(pxyz[i][j][k]) / tot for k in range(nz)] for j in range(ny)] for i in range(nx)]
    hx = _big2.entropy(_big2.marg3(p, (0,)), base)
    hy = _big2.entropy(_big2.marg3(p, (1,)), base)
    hz = _big2.entropy(_big2.marg3(p, (2,)), base)
    hxy = _big2.entropy(_big2.marg3(p, (0, 1)), base)
    hxz = _big2.entropy(_big2.marg3(p, (0, 2)), base)
    hyz = _big2.entropy(_big2.marg3(p, (1, 2)), base)
    hxyz = _big2.entropy(_big2.flat3(p), base)
    ixy = hx + hy - hxy
    ixz = hx + hz - hxz
    return RichResult(
        payload={
            "estimate": ixy - ixz,
            "ixy": ixy,
            "ixz": ixz,
            "markov_gap": hxy + hyz - hxyz - hy,
            "holds": bool(ixy - ixz >= -1e-12),
            "n": nx * ny * nz,
            "method": "Data-processing inequality I(X;Y) >= I(X;Z) -- Cover & Thomas (2006) Thm 2.8.1",
        }
    )


def cheatsheet():
    return "dpitst: Data processing inequality test"
