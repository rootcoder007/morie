# morie.fn -- function file (rootcoder007/morie)
"""Expected prediction error of least squares at a test point.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.5, book pp. 24-26 (PDF pp. 43,
45), for the model Y = X' beta + eps of equation (2.26):

    EPE(x0) = Var(y0|x0) + Var_T(yhat0) + Bias^2(yhat0)
            = sigma^2 + E_T x0' (X'X)^-1 x0 sigma^2 + 0^2            (2.27)

and, for large N with E(X) = 0 so that X'X -> N Cov(X),

    E_x0 EPE(x0) ~ sigma^2 (p/N) + sigma^2.                          (2.28)

The bias term is identically zero because the least squares estimate is
unbiased under (2.26); the book writes it as "+ 0^2" and it is returned
as such rather than being dropped.  sigma^2, when not supplied, is the
usual residual estimate RSS/(N - q) from the same design.
"""

from __future__ import annotations

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["epeols"]


def epeols(X, y, x0, sigma2=None):
    """Equations (2.27) and (2.28).

    Parameters
    ----------
    X : array-like
        N-by-p training design.  Include a constant column yourself if the
        model has an intercept; (2.26) has none.
    y : array-like
        N-vector of responses, used only to estimate sigma^2.
    x0 : array-like
        p-vector, the test point.
    sigma2 : float, optional
        Noise variance; estimated by RSS/(N - p) when omitted.

    Returns
    -------
    RichResult with keys estimate, epe, variance, bias2, sigma2, leverage,
    approx, n, p, method.
    """
    Xm = k.mat(X)
    yv = k.vec(y)
    z = k.vec(x0)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("epeols: X is empty")
    if len(yv) != n:
        raise ValueError("epeols: X and y must have the same number of rows")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("epeols: X has no columns")
    if len(z) != p:
        raise ValueError("epeols: x0 must have one entry per column of X")
    if n <= p and sigma2 is None:
        raise ValueError("epeols: need N > p to estimate sigma2")
    A = k.crossprod(Xm)
    v = k.ridgesolve(A, list(z), 0.0)
    lev = sum(z[a] * v[a] for a in range(p))
    if sigma2 is None:
        beta = k.lstsq(Xm, yv, 0.0)
        fit = [sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
        s2 = sum((yv[i] - fit[i]) ** 2 for i in range(n)) / (n - p)
    else:
        s2 = float(sigma2)
        if s2 < 0.0:
            raise ValueError("epeols: sigma2 must be non-negative")
    var = lev * s2
    bias2 = 0.0
    epe = s2 + var + bias2
    approx = s2 * (p / float(n)) + s2
    return RichResult(
        title="EPE of least squares at x0, ESL eq. (2.27)",
        summary_lines=[("n", n), ("p", p), ("EPE", epe)],
        payload={
            "estimate": epe,
            "epe": epe,
            "variance": var,
            "bias2": bias2,
            "sigma2": s2,
            "leverage": lev,
            "approx": approx,
            "n": n,
            "p": p,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.26)-(2.28)",
        },
    )


def cheatsheet():
    return "epeols: EPE(x0) = sigma^2 + x0'(X'X)^-1 x0 sigma^2, ESL eqs. (2.27)-(2.28)"
