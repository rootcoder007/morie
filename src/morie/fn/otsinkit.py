"""Adaptive iteration count for Sinkhorn given tol."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_sinkhorn_iter_count"]


def ot_sinkhorn_iter_count(a, b, C, epsilon, tol, max_iter=200):
    """
    Iterations Sinkhorn needs to reach a marginal tolerance.

    Formula: the first k with err_k < tol, over a fixed-length trace

    Verified against Cuturi (2013) Section 4.1 and Peyre & Cuturi (2019)
    Section 4.2 (Sinkhorn convergence) -- sources consulted.

    The loop is NOT stopped by the tolerance: exactly ``max_iter``
    scalings are performed, the marginal violation after each is
    recorded, and the first index below ``tol`` is read off afterwards.
    That keeps the computation a fixed-length deterministic recurrence
    while still answering the question asked.

    Parameters
    ----------
    a, b : array-like
        Non-negative marginals; closed internally.
    C : nested sequence
        Cost matrix.
    epsilon : float
        Regularisation strength.
    tol : float
        Marginal-violation threshold, in the sup norm.
    max_iter : int, optional
        Fixed trace length (default 200).

    Returns
    -------
    RichResult
        Keys: estimate (iterations needed, or ``max_iter`` if never
        reached), reached, final_error, trace, method.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn Distances. NIPS 26. Sec. 4.1.
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Sec. 4.2.
    """
    tol = float(tol)
    if not (tol > 0.0):
        raise ValueError("tol must be positive")
    trace = _big2.sinkhorn_trace(a, b, C, epsilon, max_iter)
    need = int(max_iter)
    reached = False
    for k, e in enumerate(trace):
        if e < tol:
            need = k + 1
            reached = True
            break
    return RichResult(
        payload={
            "estimate": float(need),
            "reached": reached,
            "final_error": trace[-1] if trace else float("nan"),
            "trace": trace,
            "method": "Sinkhorn iterations to reach tol, from a fixed-length trace -- Cuturi (2013) Sec. 4.1",
        }
    )


def cheatsheet():
    return "otsinkit: Adaptive iteration count for Sinkhorn given tol"
