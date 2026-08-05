# morie.fn -- function file (rootcoder007/morie)
"""Lasso regression -- an alias for :mod:`esllso`.

``ledger/wave2/DUPMAP.tsv`` records ``lassrg`` as a duplicate of
``esllso`` and it is: the same penalised least-squares problem solved by
the same cyclic coordinate descent.  Only the argument order differs.
"""

from .esllso import esl_lasso

__all__ = ["lasso_regression"]


def lasso_regression(y, X, lam, max_iter=10000, tol=1e-12):
    """Least squares with an L1 penalty, which selects while it shrinks.

    Ridge shrinks every coefficient and drops none; subset selection drops
    but does not shrink, and its objective is combinatorial.  The L1
    penalty is the convex relaxation that does both at once, and the
    non-differentiable corner at zero is exactly the feature that produces
    exact zeros rather than merely small numbers.

    Formula: ``min_beta ||y - X beta||^2 + lambda ||beta||_1`` --
    Tibshirani (1996).

    This is an alias.  The solver lives in ``morie.fn.esllso``; here the
    response comes first, as the documented signature has it.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix.
    lam : float
        Penalty, non-negative.
    max_iter : int, default 10000
        Maximum sweeps.
    tol : float, default 1e-12
        Convergence tolerance.

    Returns
    -------
    RichResult
        Whatever ``esllso.esl_lasso`` returns, unchanged.

    References
    ----------
    Tibshirani, R. (1996).  Regression shrinkage and selection via the
    lasso.  Journal of the Royal Statistical Society Series B
    58(1):267-288.  doi:10.1111/j.2517-6161.1996.tb02080.x.
    """
    return esl_lasso(X, y, lam, max_iter, tol)


def cheatsheet():
    return "lassrg: lasso regression (alias of esllso)"
