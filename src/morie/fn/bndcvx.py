# morie.fn -- function file (rootcoder007/morie)
"""Linear-programming bounds on a target parameter."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bound_convex_estimator"]


def bound_convex_estimator(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                           bounds=None):
    r"""Sharp bounds on a linear target parameter by linear
    programming, the computational engine of Mogstad, Santos and
    Torgovitsky (2018).

    The identification problem there has a common shape: the object
    of interest is a linear functional :math:`c'x` of unknown
    quantities :math:`x` (marginal-treatment-response coefficients in
    the paper), and everything the data and the maintained
    assumptions say about :math:`x` is a set of LINEAR restrictions
    -- IV-like moments as equalities, monotonicity and boundedness as
    inequalities. The identified set for the target is then exactly

    .. math:: \Big[\min_{x \in \mathcal C} c'x,\;
              \max_{x \in \mathcal C} c'x\Big],
              \quad \mathcal C = \{x: A_{ub}x \le b_{ub},\
              A_{eq}x = b_{eq},\ l \le x \le u\},

    and both ends are ordinary linear programmes. SHARPNESS is the
    point: the interval is not a bound on the identified set, it IS
    the identified set for the target, because every value between
    the two optima is attained by some feasible :math:`x` (the
    feasible set is convex and :math:`c'x` continuous).

    Infeasibility is informative rather than an error state: an empty
    :math:`\mathcal C` means the maintained assumptions contradict
    the data moments -- a specification REJECTION -- and the result
    says so. An unbounded programme means the assumptions do not
    restrict the target at all in that direction.

    Parameters
    ----------
    c : array-like, shape (k,)
        The target functional's coefficients.
    A_ub, b_ub : array-like, optional
        Inequality restrictions :math:`A_{ub} x \le b_{ub}`.
    A_eq, b_eq : array-like, optional
        Equality restrictions (typically the data moments).
    bounds : sequence of (lo, hi), optional
        Per-coordinate box constraints; defaults to [0, 1] each,
        the natural range for response probabilities.

    Returns
    -------
    RichResult
        keys: ``lower``, ``upper``, ``width``, ``argmin``, ``argmax``,
        ``feasible``, ``bounded``, ``sharp`` (True), ``k``,
        ``n_inequalities``, ``n_equalities``, ``method``.

    References
    ----------
    Mogstad, M., Santos, A. and Torgovitsky, A. (2018), "Using
    instrumental variables for inference about policy relevant
    treatment parameters", *Econometrica* 86:1589-1619, Prop. 2 and
    the linear-programming formulation of Sec. 4.
    """
    from ._sci_core import linprog

    cv = np.asarray(c, dtype=float).ravel()
    k = cv.size
    if k < 1:
        raise ValueError("the target functional needs at least one "
                         "coefficient.")
    bx = [(0.0, 1.0)] * k if bounds is None else list(bounds)
    if len(bx) != k:
        raise ValueError(f"bounds has {len(bx)} pairs for {k} coordinates.")
    kw = {}
    n_ub = n_eq = 0
    if A_ub is not None:
        kw["A_ub"] = np.atleast_2d(np.asarray(A_ub, dtype=float))
        kw["b_ub"] = np.asarray(b_ub, dtype=float).ravel()
        n_ub = kw["A_ub"].shape[0]
        if kw["A_ub"].shape[1] != k or kw["b_ub"].size != n_ub:
            raise ValueError("A_ub and b_ub have inconsistent shapes.")
    if A_eq is not None:
        kw["A_eq"] = np.atleast_2d(np.asarray(A_eq, dtype=float))
        kw["b_eq"] = np.asarray(b_eq, dtype=float).ravel()
        n_eq = kw["A_eq"].shape[0]
        if kw["A_eq"].shape[1] != k or kw["b_eq"].size != n_eq:
            raise ValueError("A_eq and b_eq have inconsistent shapes.")

    lo = linprog(cv, bounds=bx, method="highs", **kw)
    hi = linprog(-cv, bounds=bx, method="highs", **kw)
    infeasible = lo.status == 2 or hi.status == 2
    unbounded = lo.status == 3 or hi.status == 3
    return RichResult(payload={
        "lower": float(lo.fun) if lo.status == 0 else
        (-np.inf if lo.status == 3 else np.nan),
        "upper": float(-hi.fun) if hi.status == 0 else
        (np.inf if hi.status == 3 else np.nan),
        "width": (float(-hi.fun) - float(lo.fun)
                  if lo.status == 0 and hi.status == 0 else np.nan),
        "argmin": lo.x if lo.status == 0 else None,
        "argmax": hi.x if hi.status == 0 else None,
        "feasible": not infeasible, "bounded": not unbounded,
        "sharp": True,
        "sharpness_note": "the interval IS the identified set for the "
                          "target: the feasible set is convex, so every "
                          "value between the optima is attained",
        "infeasibility_note": "an empty feasible set is a specification "
                              "REJECTION -- the maintained assumptions "
                              "contradict the data moments",
        "k": int(k), "n_inequalities": int(n_ub), "n_equalities": int(n_eq),
        "method": "Sharp LP bounds on a linear target "
                  "(Mogstad-Santos-Torgovitsky 2018)"})


def cheatsheet():
    return "bndcvx: min and max of c'x over the linear restrictions -- and infeasible means REJECTED"
