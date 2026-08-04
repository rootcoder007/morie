"""Entropic-regularised OT via Sinkhorn iterations."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_sinkhorn"]


def ot_sinkhorn(a, b, C, epsilon, max_iter=200):
    """
    Entropic-regularised OT by Sinkhorn scaling.

    Formula: T = diag(u) K diag(v), K = exp(-C/eps); alternate u, v

    Verified against Cuturi (2013), *Sinkhorn Distances*, eq. (2) and
    Section 4.1 (the optimum has the form ``u_i e^{-lambda m_ij} v_j``,
    with lambda = 1/eps), and Peyre & Cuturi (2019) eq. (4.2), (4.15) --
    sources consulted.

    The loop runs exactly ``max_iter`` scalings. There is no tolerance
    and no early exit, so the result is a deterministic function of the
    inputs and both language arms agree bit for bit.

    Parameters
    ----------
    a, b : array-like
        Non-negative marginals; each closed to unit mass internally.
    C : nested sequence
        Cost matrix, ``len(a)`` by ``len(b)``.
    epsilon : float
        Regularisation strength; must be positive.
    max_iter : int, optional
        Fixed number of scalings (default 200).

    Returns
    -------
    RichResult
        Keys: estimate (the transport cost <T,C>), T, u, v, iters,
        marginal_error, method.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of
    Optimal Transport. NIPS 26, 2292-2300. Eq. (2), Sec. 4.1.
    """
    T, u, v, av, bv = _big2.sinkhorn(a, b, C, epsilon, max_iter)
    Cm = _big2.mat(C)
    cost = 0.0
    for i in range(len(T)):
        for j in range(len(T[0])):
            cost += T[i][j] * Cm[i][j]
    return RichResult(
        payload={
            "estimate": cost,
            "T": T,
            "u": u,
            "v": v,
            "iters": int(max_iter),
            "marginal_error": _big2.margerr(T, av, bv),
            "method": "Sinkhorn scaling, fixed iteration count -- Cuturi (2013) eq. (2)",
        }
    )


def cheatsheet():
    return "otsinkhorn: Entropic-regularised OT via Sinkhorn iterations"


# compact alias per ledger/NAMING.md
otsinkhorn = ot_sinkhorn
