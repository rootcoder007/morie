"""KL chain rule."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["k_l_divergence_chain"]


def k_l_divergence_chain(pxy, qxy, base=2.0):
    """
    Chain rule for relative entropy.

    Formula: D(p(x,y)||q(x,y)) = D(p(x)||q(x)) + D(p(y|x)||q(y|x))

    Verified against Cover & Thomas (2006) Theorem 2.5.3, eq. (2.67)
    p. 24, with the conditional relative entropy of eq. (2.65) --
    source consulted.

    Parameters
    ----------
    pxy, qxy : nested sequence
        Joint pmfs of the same shape; each normalised internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, marginal, conditional, residual, n, method.
        ``residual`` is joint - marginal - conditional, zero up to
        rounding whenever the chain rule applies.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.65), (2.67).
    """
    nx, ny = _big2.dims2(pxy)
    mx, my = _big2.dims2(qxy)
    if (nx, ny) != (mx, my):
        raise ValueError("pxy and qxy must have the same shape")
    pt = float(sum(float(v) for r in pxy for v in r))
    qt = float(sum(float(v) for r in qxy for v in r))
    if not (pt > 0.0 and qt > 0.0):
        raise ValueError("both pmfs must have positive total mass")
    p = [[float(pxy[i][j]) / pt for j in range(ny)] for i in range(nx)]
    q = [[float(qxy[i][j]) / qt for j in range(ny)] for i in range(nx)]
    inf = float("inf")
    joint = _big2.kldiv([v for r in p for v in r], [v for r in q for v in r], base)
    px = _big2.marg2(p, 0)
    qx = _big2.marg2(q, 0)
    marginal = _big2.kldiv(px, qx, base)
    cond = 0.0
    for i in range(nx):
        if px[i] <= 0.0:
            continue
        if qx[i] <= 0.0:
            cond = inf
            break
        term = _big2.kldiv([p[i][j] / px[i] for j in range(ny)],
                           [q[i][j] / qx[i] for j in range(ny)], base)
        if term == inf:
            cond = inf
            break
        cond += px[i] * term
    resid = float("nan") if (joint == inf or cond == inf or marginal == inf) else joint - marginal - cond
    return RichResult(
        payload={
            "estimate": joint,
            "marginal": marginal,
            "conditional": cond,
            "residual": resid,
            "n": nx * ny,
            "method": "Chain rule for relative entropy -- Cover & Thomas (2006) eq. (2.67)",
        }
    )


def cheatsheet():
    return "kchnls: KL chain rule"
