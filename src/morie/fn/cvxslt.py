# morie.fn -- function file (rootcoder007/morie)
"""Slater's condition -- Boyd & Vandenberghe Sec. 5.2.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_slater"]


def boyd_slater(f, affine=None, h=None, tol=1e-12):
    r"""Test Slater's condition: some x is STRICTLY feasible,
    :math:`f_i(x) < 0` for every non-affine :math:`f_i`, with the affine
    constraints and equalities merely satisfied.

    This is the constraint qualification that upgrades weak duality to
    STRONG duality for a convex problem. Without it the duality gap can be
    positive even when everything is convex and both problems are
    feasible -- convexity alone is not enough, which is the fact this
    function exists to keep visible.

    The refinement matters in practice: AFFINE constraints need only hold,
    not hold strictly. A problem whose feasible set is a single point
    defined by equalities can still satisfy Slater, and treating equalities
    as needing slack would wrongly declare it unqualified.

    Parameters
    ----------
    f : array-like
        Non-affine inequality constraint values at the candidate point.
    affine : array-like, optional
        Affine inequality values; these need only be :math:`\le 0`.
    h : array-like, optional
        Equality values; these need only be zero.
    tol : float
        Tolerance.

    Returns
    -------
    RichResult
        ``holds``, ``strict_margin``, ``binding`` (constraints with no
        slack), ``strong_duality_guaranteed``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A strictly interior point qualifies.

    >>> r = boyd_slater([-1.0, -0.5])
    >>> bool(r["holds"]), round(r["strict_margin"], 3)
    (True, 0.5)

    A point ON the boundary of a non-affine constraint does not: strict
    inequality is the requirement.

    >>> bool(boyd_slater([-1.0, 0.0])["holds"])
    False

    Affine constraints need only HOLD, not hold strictly -- so a tight
    affine constraint still qualifies, which is the refinement that keeps
    equality-defined problems from being wrongly disqualified.

    >>> bool(boyd_slater([-1.0], affine=[0.0], h=[0.0])["holds"])
    True

    Without Slater, strong duality is not guaranteed even for a convex
    problem, and the result says so rather than implying a gap exists.

    >>> boyd_slater([0.0])["strong_duality_guaranteed"]
    False
    """
    fv = np.atleast_1d(np.asarray(f, dtype=float)).ravel() if f is not None else np.zeros(0)
    av = np.atleast_1d(np.asarray(affine, dtype=float)).ravel() if affine is not None else np.zeros(0)
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel() if h is not None else np.zeros(0)
    strict = bool(np.all(fv < -tol)) if fv.size else True
    aff_ok = bool(np.all(av <= tol)) if av.size else True
    eq_ok = bool(np.all(np.abs(hv) <= 1e-08)) if hv.size else True
    holds = bool(strict and aff_ok and eq_ok)
    margin = float(-fv.max()) if fv.size else float("inf")
    return RichResult(
        title="Slater's condition",
        summary_lines=[("holds", holds), ("strict margin", margin),
                       ("non-affine constraints", int(fv.size))],
        warnings=[] if holds else
        ["Slater's condition fails, so strong duality is not guaranteed; "
         "a convex problem can still have a positive duality gap"],
        payload={
            "holds": holds, "strict_margin": margin,
            "binding": np.flatnonzero(fv >= -tol) if fv.size else np.zeros(0, dtype=int),
            "strong_duality_guaranteed": holds,
            "strict_inequalities_ok": strict, "affine_ok": aff_ok,
            "equalities_ok": eq_ok, "method": "boyd_slater",
        },
    )


def cheatsheet():
    return "cvxslt: convexity alone does NOT give strong duality; affine constraints need only hold"


# compact alias per ledger/NAMING.md
boydslater = boyd_slater
