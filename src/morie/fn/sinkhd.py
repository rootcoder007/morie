"""Sinkhorn divergence (entropic OT)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["sinkhorn_distance"]


def sinkhorn_distance(a, b, C, eps, max_iter=200):
    """
    Dual-Sinkhorn divergence between two discrete measures.

    Formula: <P^lambda, C> where P^lambda = argmin <P,C> - (1/lambda) h(P)

    Verified against Cuturi (2013), Definition 1 and eq. (2) -- source
    consulted. Cuturi parameterises by lambda; here ``eps = 1/lambda``,
    so ``K = exp(-C/eps) = exp(-lambda C)``. Both the transport cost
    (Cuturi's ``d^lambda_M``) and the regularised objective are
    returned, because they are different numbers and are often
    confused.

    Parameters
    ----------
    a, b : array-like
        Non-negative marginals; closed internally.
    C : nested sequence
        Cost matrix.
    eps : float
        Regularisation strength, ``1/lambda``.
    max_iter : int, optional
        Fixed number of Sinkhorn scalings (default 200).

    Returns
    -------
    RichResult
        Keys: estimate (the transport cost), objective, entropy,
        lambda_, iters, method.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of
    Optimal Transport. NIPS 26, 2292-2300. Definition 1, eq. (2).
    """
    epsv = float(eps)
    if not (epsv > 0.0):
        raise ValueError("eps must be positive")
    T, u, v, av, bv = _big2.sinkhorn(a, b, C, epsv, max_iter)
    Cm = _big2.mat(C)
    cost = 0.0
    h = 0.0
    for i in range(len(T)):
        for j in range(len(T[0])):
            cost += T[i][j] * Cm[i][j]
            t = T[i][j]
            if t > 0.0:
                h -= t * (float(np.log(t)) - 1.0)
    return RichResult(
        payload={
            "estimate": cost,
            "objective": cost - epsv * h,
            "entropy": h,
            "lambda_": 1.0 / epsv,
            "iters": int(max_iter),
            "method": "Dual-Sinkhorn divergence <P,C> -- Cuturi (2013) Def. 1, eq. (2)",
        }
    )


def cheatsheet():
    return "sinkhd: Sinkhorn divergence (entropic OT)"
