# morie.fn -- function file (rootcoder007/morie)
"""Population least-squares coefficient from second moments.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.4, book pp. 18-19 (PDF pp. 37-38).

    EPE(f) = E(Y - f(X))^2                                          (2.9)
    beta   = [E(X X')]^-1 E(X Y)                                    (2.16)

Equation (2.16) is what you get by plugging the linear model (2.15) into
the expected prediction error (2.9) and differentiating.  It is NOT the
same object as the sample formula (2.6): (2.16) is stated in terms of
the population moments E(XX') and E(XY), which this function estimates
by the sample moment matrices (1/N) X'X and (1/N) X'y.  As the book says
on p. 19, "the least squares solution (2.6) amounts to replacing the
expectation in (2.16) by averages over the training data" -- so with the
same design the two agree exactly, which is the anchor used for it.

No intercept is added: (2.16) has none.  Put a constant column in X if
you want one, exactly as the book does throughout Chapter 2.
"""

from __future__ import annotations

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["epetheor"]


def epetheor(X, y):
    """Solve beta = [E(XX')]^-1 E(XY) of equation (2.16).

    Parameters
    ----------
    X : array-like
        N-by-p matrix of inputs.
    y : array-like
        N-vector of responses.

    Returns
    -------
    RichResult with keys estimate, beta, exx, exy, epe, eyy, n, p, method.
    """
    Xm = k.mat(X)
    yv = k.vec(y)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("epetheor: X is empty")
    if len(yv) != n:
        raise ValueError("epetheor: X and y must have the same number of rows")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("epetheor: X has no columns")
    exx = [[sum(Xm[i][a] * Xm[i][b] for i in range(n)) / n for b in range(p)] for a in range(p)]
    exy = [sum(Xm[i][a] * yv[i] for i in range(n)) / n for a in range(p)]
    beta = k.ridgesolve(exx, exy, 0.0)
    pred = [sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
    epe = sum((yv[i] - pred[i]) ** 2 for i in range(n)) / n
    eyy = sum(v * v for v in yv) / n
    return RichResult(
        title="Population least squares, ESL eq. (2.16)",
        summary_lines=[("n", n), ("p", p), ("EPE", epe)],
        payload={
            "estimate": beta[0],
            "beta": beta,
            "exx": exx,
            "exy": exy,
            "epe": epe,
            "eyy": eyy,
            "n": n,
            "p": p,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.9), (2.15), (2.16)",
        },
    )


def cheatsheet():
    return "epetheor: beta = [E(XX')]^-1 E(XY), ESL eq. (2.16), with EPE (2.9)"
