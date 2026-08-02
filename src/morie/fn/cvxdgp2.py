# morie.fn -- function file (rootcoder007/morie)
"""Duality gap -- Boyd & Vandenberghe Sec. 5.2.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_duality_gap"]


def boyd_duality_gap(primal, dual, tol=1e-09):
    r"""The gap :math:`f_0(x) - g(\lambda, \nu)` between a feasible primal
    value and a dual value.

    The gap is the certificate. Any feasible x and any
    :math:`\lambda \ge 0` together prove
    :math:`p^\star \in [g, f_0]`, so a small gap means the primal point is
    provably near-optimal WITHOUT knowing the optimum. That is what makes
    it the natural stopping rule for an interior-point method.

    A negative gap is impossible under weak duality, so it means the inputs
    are wrong -- an infeasible x, or a multiplier with the wrong sign -- and
    is reported as such rather than returned as a small number.

    Parameters
    ----------
    primal : float
        Objective value at a FEASIBLE primal point.
    dual : float
        Dual function value at a feasible dual point.
    tol : float
        Tolerance below which the gap counts as closed.

    Returns
    -------
    RichResult
        ``gap``, ``relative_gap``, ``closed``, ``strong_duality``,
        ``bracket``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> r = boyd_duality_gap(5.0, 4.0)
    >>> r["gap"]
    1.0

    The bracket is the certificate: the optimum lies inside it.

    >>> r["bracket"]
    (4.0, 5.0)

    A closed gap means strong duality holds AT THESE POINTS and both are
    optimal.

    >>> boyd_duality_gap(2.0, 2.0)["strong_duality"]
    True

    A negative gap violates weak duality and is an input error, not a
    small number.

    >>> boyd_duality_gap(1.0, 3.0)
    Traceback (most recent call last):
        ...
    ValueError: dual exceeds primal, which violates weak duality: check feasibility and the sign of lambda
    """
    p = float(primal)
    d = float(dual)
    if not (np.isfinite(p) and np.isfinite(d)):
        raise ValueError("primal and dual must both be finite")
    gap = p - d
    if gap < -abs(tol):
        raise ValueError(
            "dual exceeds primal, which violates weak duality: check "
            "feasibility and the sign of lambda")
    gap = max(gap, 0.0)
    rel = gap / max(abs(p), 1e-300)
    closed = bool(gap <= tol)
    return RichResult(
        title="Duality gap",
        summary_lines=[("primal", p), ("dual", d), ("gap", gap),
                       ("relative", rel)],
        payload={
            "gap": gap, "relative_gap": rel, "closed": closed,
            "strong_duality": closed, "bracket": (d, p),
            "primal": p, "dual": d,
            "method": "boyd_duality_gap",
        },
    )


def cheatsheet():
    return "cvxdgp2: gap CERTIFIES near-optimality without knowing p*; a negative gap is an input error"
