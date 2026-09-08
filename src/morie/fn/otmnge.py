"""Negative entropy of a transport plan."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_marginal_negent"]


def ot_marginal_negent(T):
    """
    Discrete entropy of a coupling matrix.

    Formula: H(T) = -sum T_ij (log T_ij - 1)

    Verified against Peyre & Cuturi (2019), *Computational Optimal
    Transport*, eq. (4.1) -- source consulted. Note the ``- 1``: this is
    the OT convention, one nat per unit mass larger than the Shannon
    entropy, and it must not be "simplified" away.

    Parameters
    ----------
    T : nested sequence
        Non-negative coupling matrix.

    Returns
    -------
    RichResult
        Keys: estimate, shannon, mass, nrow, ncol, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport.
    Foundations and Trends in Machine Learning 11(5-6). Eq. (4.1).
    """
    m = _big2.mat(T)
    nr, nc = len(m), len(m[0])
    h = 0.0
    sh = 0.0
    mass = 0.0
    for r in m:
        for v in r:
            if v < 0.0:
                raise ValueError("T must be non-negative")
            mass += v
            if v > 0.0:
                lv = float(np.log(v))
                h -= v * (lv - 1.0)
                sh -= v * lv
            else:
                h += 0.0
    return RichResult(
        payload={
            "estimate": h,
            "shannon": sh,
            "mass": mass,
            "nrow": nr,
            "ncol": nc,
            "method": "Discrete entropy H(T) = -sum T(log T - 1) -- Peyre & Cuturi (2019) eq. (4.1)",
        }
    )


def cheatsheet():
    return "otmnge: Negative entropy of a transport plan"
