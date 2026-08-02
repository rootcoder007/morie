# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Truncated power basis (ESL Ch 5.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_basis_truncated"]


def esl_basis_truncated(x, knots, p=3):
    """
    Truncated power basis: 1, x, ..., x^p, then (x - xi_k)_+^p per knot.

    This is the textbook basis for a degree-p spline and it is the
    clearest way to SEE what a spline is: a polynomial plus one extra
    term per knot that switches on only to the right of that knot.
    ESL Ch 5.2 introduces it for exactly that reason and then warns
    against computing with it — powers of x and of (x - xi) are
    strongly collinear, so the basis is badly conditioned and the
    normal equations lose accuracy. B-splines are the numerically
    stable equivalent. The condition number is reported here so that
    warning is measurable rather than folklore.

    Column order: [1, x, ..., x^p, (x-xi_1)_+^p, ...], row-major.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    knots : array-like
        Interior knot locations (may be empty).
    p : int
        Degree, >= 0. p = 3 gives the usual cubic spline.

    Returns
    -------
    result : dict
        Keys: estimate (number of basis functions), basis (row-major
        n x M), n_basis, degree, knots, condition_number, n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 5.2 (Eq. 5.3-5.5).

    Examples
    --------
    One knot at 1, cubic: four polynomial columns plus one hinge that
    is zero until x passes the knot.

    >>> out = esl_basis_truncated([0.0, 1.0, 2.0], [1.0], 3)
    >>> out["n_basis"]
    5
    >>> out["basis"][:5]           # row for x = 0
    [1.0, 0.0, 0.0, 0.0, 0.0]
    >>> out["basis"][5:10]         # row for x = 1: hinge still zero
    [1.0, 1.0, 1.0, 1.0, 0.0]
    >>> out["basis"][10:]          # row for x = 2: hinge active
    [1.0, 2.0, 4.0, 8.0, 1.0]
    >>> esl_basis_truncated([0.0, 1.0], [], 1)["n_basis"]
    2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    knots = np.atleast_1d(np.asarray(knots, dtype=float)) if np.size(knots) else np.array([])
    p = int(p)
    if p < 0:
        raise ValueError(f"the degree must be >= 0; got {p}.")
    if knots.size and np.any(np.diff(np.sort(knots)) == 0):
        raise ValueError("duplicate knots make the basis rank deficient.")
    cols = [x ** j for j in range(p + 1)]
    for k in np.sort(knots):
        cols.append(np.where(x > k, (x - k) ** p, 0.0))
    B = np.column_stack(cols)
    try:
        cond = float(np.linalg.cond(B))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return RichResult(payload={
        "estimate": int(B.shape[1]),
        "basis": [float(v) for v in B.ravel()],
        "n_basis": int(B.shape[1]), "degree": p,
        "knots": [float(v) for v in np.sort(knots)],
        "condition_number": cond, "n": int(x.size),
        "method": "truncated power basis; ill-conditioned by construction, B-splines preferred"})


def cheatsheet():
    return "esltrs: [1,x,..,x^p, (x-xi)_+^p]; clear to read, bad to compute with"
