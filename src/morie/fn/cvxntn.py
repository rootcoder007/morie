# morie.fn -- function file (rootcoder007/morie)
"""Newton step for unconstrained minimisation -- Boyd & Vandenberghe (2004) Sec 9.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_newton"]


def boyd_newton(grad, hess, ridge=0.0):
    r"""The Newton step :math:`\Delta x_{nt} = -\nabla^2 f(x)^{-1}\nabla f(x)`.

    Solved as a linear system rather than by forming the inverse, which is
    both faster and better conditioned. When the Hessian is positive definite
    a Cholesky factorisation is used; otherwise the routine falls back to a
    least-squares solve and says so in ``warnings``, because away from a
    minimum the Newton "step" need not be a descent direction at all.

    Parameters
    ----------
    grad : array-like
        Gradient :math:`\nabla f(x)`, shape ``(n,)``.
    hess : array-like
        Hessian :math:`\nabla^2 f(x)`, shape ``(n, n)``. Must be symmetric.
    ridge : float
        Optional Levenberg-style shift added to the diagonal, which makes an
        indefinite Hessian usable. Default 0 (pure Newton).

    Returns
    -------
    RichResult
        ``step`` (the Newton direction), ``decrement`` (see
        :func:`~morie.fn.cvxnda.boyd_newton_decrement`), ``is_descent``, and
        ``pd`` recording whether the Hessian was positive definite.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    On a quadratic, one Newton step lands exactly on the minimiser.

    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> x = np.array([5.0, -2.0])          # f(x) = 0.5 x'Ax, minimiser at 0
    >>> step = boyd_newton(A @ x, A)["step"]
    >>> [float(round(v, 12)) + 0.0 for v in x + step]
    [0.0, 0.0]

    The direction is a descent direction when the Hessian is positive
    definite.

    >>> bool(boyd_newton(A @ x, A)["is_descent"])
    True

    An indefinite Hessian is reported rather than silently trusted.

    >>> bool(boyd_newton([1.0], [[-1.0]])["pd"])
    False
    """
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    Hm = np.atleast_2d(np.asarray(hess, dtype=float))
    n = g.size
    if Hm.shape != (n, n):
        raise ValueError(f"hess must be ({n}, {n}) to match grad, got {Hm.shape}")
    if not np.allclose(Hm, Hm.T, atol=1e-10):
        raise ValueError("hess must be symmetric")
    if ridge:
        Hm = Hm + ridge * np.eye(n)
    warnings_list = []
    try:
        L = np.linalg.cholesky(Hm)
        pd = True
        step = -np.linalg.solve(Hm, g)
        del L
    except np.linalg.LinAlgError:
        pd = False
        warnings_list.append(
            "Hessian is not positive definite; using a least-squares solve. "
            "The result need not be a descent direction -- consider `ridge`."
        )
        step = -np.linalg.lstsq(Hm, g, rcond=None)[0]
    decrement = float(np.sqrt(max(g @ -step, 0.0)))
    return RichResult(
        title="Newton step",
        summary_lines=[("n", int(n)), ("Hessian PD", bool(pd)), ("decrement", decrement)],
        warnings=warnings_list,
        payload={
            "step": step,
            "decrement": decrement,
            "is_descent": bool(g @ step < 0),
            "pd": bool(pd),
            "method": "boyd_newton",
        },
    )


def cheatsheet():
    return "cvxntn: Newton step by linear solve, not inversion; reports whether the Hessian was PD"
