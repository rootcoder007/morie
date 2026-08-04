# morie.fn -- function file (rootcoder007/morie)
"""Elastic net penalized residual sum of squares."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['enetobj', 'elastic_net_objective']


def enetobj(X, y, beta, lam, alpha, add_intercept=True):
    """Elastic net penalized residual sum of squares.

    Formula: PRSS(beta, lambda, alpha) = RSS(beta) + lambda [ (1 - alpha)/2 * sum_j b_j^2 + alpha * sum_j |b_j| ]

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    beta : array-like
        Coefficient vector.
    lam : float
        Regularization parameter lambda; must be non-negative.
    alpha : float
        Mixing parameter in [0, 1]: 0 is the ridge penalty, 1 is the lasso penalty.
    add_intercept : bool
        Treat the first entry of beta as an unpenalized intercept and prepend a column of ones to X.

    Returns
    -------
    RichResult
        ``prss``, ``rss``, ``penalty``, ``l1``, ``l2``, ``lambda``, ``alpha``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.6.2 p. 94 describes the elastic net as the combination of the ridge and lasso penalties implemented by glmnet, and Chapter 7 p. 230 defines its mixing parameter alpha: alpha = 0 gives the ridge penalty, alpha = 1 the lasso, and values in between the elastic net.  The book names those two endpoints but does not print the mixed objective itself; the (1 - alpha)/2 and alpha weights are the glmnet parameterization of Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear Models via Coordinate Descent, Journal of Statistical Software 33(1):1-22, doi:10.18637/jss.v033.i01, which is the package the book cites for this penalty.  Both chapters read from the PDFs; the mixing weights come from the cited paper, and the endpoints agree with the book: alpha = 0 reproduces MVSML Sect. 3.6.1 up to the factor 1/2 on the ridge term, alpha = 1 reproduces Sect. 3.6.2 exactly.
    """
    Xm = C.mat(X)
    if add_intercept:
        Xm = C.cbind1(Xm)
    y = C.vec(y)
    b = C.vec(beta)
    lam = float(lam)
    a = float(alpha)
    n, p = len(Xm), len(Xm[0])
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    if len(b) != p:
        raise ValueError("beta must have one entry per column of the design")
    if lam < 0.0:
        raise ValueError("lambda must be non-negative")
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    rss = sum((y[i] - sum(Xm[i][j] * b[j] for j in range(p))) ** 2
              for i in range(n))
    start = 1 if add_intercept else 0
    l2 = sum(b[j] * b[j] for j in range(start, p))
    l1 = sum(abs(b[j]) for j in range(start, p))
    pen = lam * (0.5 * (1.0 - a) * l2 + a * l1)
    return RichResult(payload={
        "prss": rss + pen, "rss": rss, "penalty": pen, "l1": l1, "l2": l2,
        "lambda": lam, "alpha": a, "n": n, "p": p,
        "method": "Elastic net penalized RSS, MVSML Sect. 3.6.2 / Chap. 7"})


elastic_net_objective = enetobj


def cheatsheet():
    return 'enobj: Elastic net penalized residual sum of squares.'
