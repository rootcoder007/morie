# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Frechet-Hoeffding bounds on a joint distribution.

Frechet (1951), "Sur les tableaux de correlation dont les marges sont
donnees", Annales de l'Universite de Lyon, Section A, 14:53-77, and
Hoeffding (1940), "Masstabinvariante Korrelationstheorie", Schriften
des Mathematischen Instituts der Universitat Berlin 5:181-233.  For
any joint distribution with the given margins,

    W(u, v) = max(u + v - 1, 0)  <=  C(u, v)  <=  min(u, v) = M(u, v),

and both bounds are attained: M by comonotone variables, W by
countermonotone ones (W is a copula only in two dimensions).  Both
bounds are themselves copulas here, so they satisfy the boundary
conditions C(u, 1) = u and C(1, v) = v exactly -- which is what the
tests check, along with the bracketing of the independence copula uv.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["frechet_hoeffding_bounds"]


def frechet_hoeffding_bounds(F_0, F_1, joint=None):
    """Lower and upper bounds at each pair of marginal probabilities.

    Parameters
    ----------
    F_0, F_1 : array-like
        Marginal CDF values in [0, 1], of equal length.
    joint : array-like, optional
        A candidate joint CDF at the same points; when supplied the
        result reports whether it respects the bounds.
    """
    u = core.vec(F_0)
    v = core.vec(F_1)
    if len(u) == 0:
        raise ValueError("frechet_hoeffding_bounds: F_0 is empty")
    if len(v) != len(u):
        raise ValueError("frechet_hoeffding_bounds: F_0 and F_1 have different lengths")
    for a in list(u) + list(v):
        if a < 0 or a > 1:
            raise ValueError("frechet_hoeffding_bounds: marginal probabilities must lie in [0, 1]")
    lo = [max(u[i] + v[i] - 1.0, 0.0) for i in range(len(u))]
    hi = [min(u[i], v[i]) for i in range(len(u))]
    ind = [u[i] * v[i] for i in range(len(u))]
    width = [hi[i] - lo[i] for i in range(len(u))]
    if joint is None:
        jv = None
        ok = 1
        viol = 0
    else:
        jv = core.vec(joint)
        if len(jv) != len(u):
            raise ValueError("frechet_hoeffding_bounds: joint and F_0 have different lengths")
        viol = sum(1 for i in range(len(u)) if jv[i] < lo[i] - 1e-12 or jv[i] > hi[i] + 1e-12)
        ok = 1 if viol == 0 else 0
    return RichResult(
        title="Frechet-Hoeffding bounds",
        summary_lines=[("points", len(u))],
        payload={
            "estimate": hi[0],
            "lower": lo,
            "upper": hi,
            "independence": ind,
            "width": width,
            "respects_bounds": ok,
            "n_violations": viol,
            "n": len(u),
            "method": "max(u + v - 1, 0) <= C(u, v) <= min(u, v), Frechet (1951); Hoeffding (1940)",
        },
    )


def cheatsheet():
    return "frdbnd: Frechet-Hoeffding bounds"
