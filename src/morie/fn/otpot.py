"""Recover log potentials f,g from Sinkhorn marginals."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_pot_log_potentials"]


def ot_pot_log_potentials(u, v, epsilon):
    """
    Dual potentials from Sinkhorn scalings.

    Formula: u = exp(f/eps), v = exp(g/eps), so f = eps log u

    Verified against Peyre & Cuturi (2019), eq. (4.30)-(4.31) --
    source consulted: "(u, v) = (e^{f/eps}, e^{g/eps}): Sinkhorn
    scalings".

    Parameters
    ----------
    u, v : array-like
        Positive Sinkhorn scaling vectors.
    epsilon : float
        Regularisation strength.

    Returns
    -------
    RichResult
        Keys: estimate (``f``), g, epsilon, shift, method. ``shift`` is
        ``mean(f)``: the dual pair is only determined up to a constant
        that moves between ``f`` and ``g``, and reporting it keeps that
        indeterminacy visible.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    eq. (4.30)-(4.31).
    """
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be positive")
    uv = [float(t) for t in np.atleast_1d(np.asarray(u, dtype=float))]
    vv = [float(t) for t in np.atleast_1d(np.asarray(v, dtype=float))]
    if min(uv) <= 0.0 or min(vv) <= 0.0:
        raise ValueError("scalings must be strictly positive")
    f = [eps * float(np.log(t)) for t in uv]
    g = [eps * float(np.log(t)) for t in vv]
    return RichResult(
        payload={
            "estimate": f,
            "g": g,
            "epsilon": eps,
            "shift": sum(f) / len(f),
            "method": "Dual potentials f = eps log u -- Peyre & Cuturi (2019) eq. (4.31)",
        }
    )


def cheatsheet():
    return "otpot: Recover log potentials f,g from Sinkhorn marginals"
