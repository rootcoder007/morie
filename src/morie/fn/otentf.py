"""Free energy of an OT plan = primal - dual."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_free_energy"]


def ot_free_energy(T, C, a, b, f, g, epsilon):
    """
    Primal-dual gap of the entropic OT problem.

    Formula: F = <T,C> - eps H(T) - <a,f> - <b,g>

    Verified against Peyre & Cuturi (2019), eq. (4.30)-(4.32) -- source
    consulted. The dual objective is
    ``<f,a> + <g,b> - eps <e^{f/eps}, K e^{g/eps}>``, and at the optimum
    it equals the primal ``<T,C> - eps H(T)``; the difference returned
    here is therefore zero at optimality and positive otherwise.

    Parameters
    ----------
    T : nested sequence
        Coupling matrix.
    C : nested sequence
        Cost matrix, same shape.
    a, b : array-like
        Marginals; closed internally.
    f, g : array-like
        Dual potentials, in the same units as ``C``.
    epsilon : float
        Regularisation strength.

    Returns
    -------
    RichResult
        Keys: estimate (the gap), primal, dual_pairing, entropy,
        epsilon, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    eq. (4.30)-(4.32).
    """
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be positive")
    Tm = _big2.mat(T)
    Cm = _big2.mat(C)
    if len(Tm) != len(Cm) or len(Tm[0]) != len(Cm[0]):
        raise ValueError("T and C must have the same shape")
    av = [float(v) for v in _big2.pnorm(np.atleast_1d(np.asarray(a, dtype=float)))]
    bv = [float(v) for v in _big2.pnorm(np.atleast_1d(np.asarray(b, dtype=float)))]
    fv = [float(v) for v in np.atleast_1d(np.asarray(f, dtype=float))]
    gv = [float(v) for v in np.atleast_1d(np.asarray(g, dtype=float))]
    if len(av) != len(Tm) or len(bv) != len(Tm[0]):
        raise ValueError("marginals do not match the shape of T")
    if len(fv) != len(av) or len(gv) != len(bv):
        raise ValueError("potentials do not match the marginals")
    cost = 0.0
    h = 0.0
    for i in range(len(Tm)):
        for j in range(len(Tm[0])):
            t = Tm[i][j]
            cost += t * Cm[i][j]
            if t > 0.0:
                h -= t * (float(np.log(t)) - 1.0)
    pair = sum(av[i] * fv[i] for i in range(len(av))) + sum(bv[j] * gv[j] for j in range(len(bv)))
    primal = cost - eps * h
    return RichResult(
        payload={
            "estimate": primal - pair,
            "primal": primal,
            "dual_pairing": pair,
            "entropy": h,
            "epsilon": eps,
            "method": "Entropic OT primal-dual gap -- Peyre & Cuturi (2019) eq. (4.30)-(4.32)",
        }
    )


def cheatsheet():
    return "otentf: Free energy of an OT plan = primal - dual"


# compact alias per ledger/NAMING.md
otfreeenergy = ot_free_energy
