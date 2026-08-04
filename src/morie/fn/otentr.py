"""Entropic regulariser term in entropic OT objective."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_entropy_regulariser"]


def ot_entropy_regulariser(T, epsilon):
    """
    Entropic regularisation term of a coupling.

    Formula: eps * H(T) = -eps * sum T_ij (log T_ij - 1)

    Verified against Peyre & Cuturi (2019) eq. (4.1)-(4.2) and Cuturi
    (2013) eq. (2) -- sources consulted. The regularised OT problem is
    ``min <P, C> - eps H(P)``, so this returns the quantity that is
    subtracted.

    Parameters
    ----------
    T : nested sequence
        Non-negative coupling matrix.
    epsilon : float
        Regularisation strength; must be positive.

    Returns
    -------
    RichResult
        Keys: estimate, entropy, epsilon, n, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport.
    Eq. (4.1), (4.2). Cuturi, M. (2013), NIPS 26, eq. (2).
    """
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be positive")
    m = _big2.mat(T)
    h = 0.0
    for r in m:
        for v in r:
            if v < 0.0:
                raise ValueError("T must be non-negative")
            if v > 0.0:
                h -= v * (float(np.log(v)) - 1.0)
    return RichResult(
        payload={
            "estimate": eps * h,
            "entropy": h,
            "epsilon": eps,
            "n": len(m) * len(m[0]),
            "method": "Entropic regulariser eps*H(T) -- Peyre & Cuturi (2019) eq. (4.2)",
        }
    )


def cheatsheet():
    return "otentr: Entropic regulariser term in entropic OT objective"
