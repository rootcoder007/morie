# morie.fn -- function file (rootcoder007/morie)
"""Dual of entropically regularised optimal transport."""

import math

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_regularised_dual"]


def ot_regularised_dual(a, b, C, epsilon, max_iter=200):
    """Dual potentials of the entropic transport problem.

    Entropic smoothing turns a constrained linear program into an
    unconstrained smooth concave maximisation in the two potentials, and
    that is what makes stochastic and large-scale solvers possible.  The
    potentials are computed in the log domain, so the result survives
    small ``epsilon`` where the exponentiated form underflows.

    Formula: ``max_{f,g} <a,f> + <b,g> - eps sum_ij exp((f_i + g_j -
    C_ij)/eps)`` -- Genevay et al. (2016) eq. (2); Peyre & Cuturi (2019)
    eq. (4.30).  At the optimum the exponential sum is the total mass, so
    the dual value reduces to ``<a,f> + <b,g> - eps``.

    Parameters
    ----------
    a, b : array-like
        Marginals, equal total mass.
    C : array-like, shape (n, m)
        Ground cost.
    epsilon : float
        Regularisation strength, positive.
    max_iter : int, default 200
        Sinkhorn sweeps.  Fixed count.

    Returns
    -------
    RichResult
        ``f``, ``g``, ``dual_value``, ``primal_cost``, ``n``, ``m``.

    References
    ----------
    Genevay, A., Cuturi, M., Peyre, G. and Bach, F. (2016).  Stochastic
    optimization for large-scale optimal transport.  Advances in Neural
    Information Processing Systems 29:3440-3448.
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cm = core.mat(C)
    n, m = len(aa), len(bb)
    if len(Cm) != n or len(Cm[0]) != m:
        raise ValueError("cost matrix does not match the marginals")
    T, f, g = ot.sinkhorn(aa, bb, Cm, float(epsilon), max_iter)
    tot = sum(T[i][j] for i in range(n) for j in range(m))
    dual = (sum(aa[i] * f[i] for i in range(n) if aa[i] > 0.0)
            + sum(bb[j] * g[j] for j in range(m) if bb[j] > 0.0)
            - float(epsilon) * tot)
    return RichResult(payload={
        "f": f, "g": g, "dual_value": dual,
        "primal_cost": ot.frob(T, Cm), "n": n, "m": m,
        "method": "Entropic optimal transport dual"})


def cheatsheet():
    return "otreg: dual potentials of entropically regularised transport"
