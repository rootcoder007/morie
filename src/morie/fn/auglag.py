# morie.fn -- function file (rootcoder007/morie)
"""Augmented Lagrangian for equality-constrained minimisation."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["auglag", "augmented_lagrangian"]


def auglag(f, g, x, lam=None, mu=1.0):
    """Value, constraint violation and multiplier update of L_A.

    A quadratic penalty on the constraints alone only reaches the
    constrained optimum as the penalty grows without bound, which makes
    the subproblems ill-conditioned.  Adding an explicit multiplier term
    removes that requirement:

        L_A(x, lambda, mu) = f(x) + sum_i lambda_i g_i(x)
                             + (mu/2) sum_i g_i(x)^2

    and, after each (approximate) minimisation over x, the multipliers are
    refreshed by the first-order rule

        lambda_i <- lambda_i + mu g_i(x),

    which converges for a finite mu.  This routine evaluates L_A at a
    given point and returns the updated multipliers; the outer search is
    the caller's, which keeps the routine free of any tolerance.

    Parameters
    ----------
    f : callable
        Objective, mapping x to a scalar.
    g : callable
        Constraint map, returning the vector of g_i(x) that should be
        zero at a feasible point.
    x : array-like
        Point at which to evaluate.
    lam : array-like or None
        Current multipliers; ``None`` starts them at zero.
    mu : float
        Penalty parameter, strictly positive.

    Returns
    -------
    RichResult
        ``value``, ``objective``, ``linear``, ``penalty``, ``violation``,
        ``lambda``, ``mu``, ``m``.

    References
    ----------
    Hestenes, M. R. (1969), "Multiplier and gradient methods", Journal of
    Optimization Theory and Applications 4(5), 303-320, and Powell,
    M. J. D. (1969), "A method for nonlinear constraints in minimization
    problems", in Fletcher, R. (ed.), Optimization, Academic Press,
    283-298, which independently introduced the multiplier term and the
    update lambda <- lambda + mu g(x).  Standard published form; neither
    source was in the local corpus and neither was read for this
    implementation.
    """
    x = C.vec(x)
    gv = C.vec(g(x))
    m = len(gv)
    if lam is None:
        lm = [0.0] * m
    else:
        lm = C.vec(lam)
        if len(lm) != m:
            raise ValueError("lam must have one entry per constraint")
    mu = float(mu)
    if mu <= 0.0:
        raise ValueError("mu must be strictly positive")
    fv = float(f(x))
    lin = sum(lm[i] * gv[i] for i in range(m))
    pen = 0.5 * mu * sum(v * v for v in gv)
    return RichResult(payload={
        "value": fv + lin + pen, "objective": fv, "linear": lin,
        "penalty": pen, "violation": max((abs(v) for v in gv), default=0.0),
        "lambda": [lm[i] + mu * gv[i] for i in range(m)], "mu": mu,
        "m": m,
        "method": "Augmented Lagrangian (Hestenes 1969; Powell 1969)"})


augmented_lagrangian = auglag


def cheatsheet():
    return "auglag: Augmented Lagrangian for equality-constrained minimisation."
