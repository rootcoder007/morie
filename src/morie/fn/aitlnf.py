# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood fit of the additive logistic normal distribution."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lognormfit', 'logistic_normal_fit']


def lognormfit(X, ref=None, ddof=1):
    """Maximum likelihood fit of the additive logistic normal distribution.

    Formula: muhat = mean of alr(x_r);  Sigmahat = sample covariance of alr(x_r)

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; all parts strictly positive.
    ref : int
        1-based index of the reference part for the alr transform; the default D uses the last part.
    ddof : int
        Divisor correction for the covariance: 1 gives the unbiased n - 1 divisor, 0 gives the maximum likelihood n divisor.

    Returns
    -------
    RichResult
        ``mu``, ``Sigma``, ``ref``, ``loglik``, ``n``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The log-ratio algebra and the additive logistic normal law were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sects. 4.1 and 4.3, which attribute the law to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Because a composition is additive logistic normal exactly when its alr transform is multivariate normal, and the alr map is a bijection that does not depend on the parameters, the maximum likelihood estimates of mu and Sigma are the ordinary multivariate normal estimates computed on the transformed data.  The reported ``loglik`` is the log-likelihood on the SIMPLEX, so it includes the sum of the log-Jacobians -sum_r sum_i log x_ri; Sect. 4.3 eq (15) prints the classical logistic normal density in ilr coordinates with Jacobian (sqrt(D) x_1 x_2 ... x_D)^-1.  In the alr coordinates used here the sqrt(D) contributed by the ilr basis is absent, giving (x_1 x_2 ... x_D)^-1.  That factor was re-derived rather than assumed: with y_i = log(x_i/x_D) and free coordinates x_1..x_{D-1}, dy/dx = diag(1/x_i) + (1/x_D) 1 1', whose determinant is (prod_{i<D} 1/x_i)(1 + (1 - x_D)/x_D) = 1 / prod_{i=1}^{D} x_i.  With ``ddof`` = 1 the covariance is the unbiased estimate rather than the MLE, so ``loglik`` is then not the maximised value.
    """
    Xm = C.mat(X)
    n = len(Xm)
    if n < 2:
        raise ValueError("the fit needs at least two compositions")
    D = len(Xm[0])
    if D < 2:
        raise ValueError("the logistic normal needs at least two parts")
    for row in Xm:
        if any(v <= 0.0 for v in row):
            raise ValueError("compositions must be strictly positive")
    k = D if ref is None else int(ref)
    if not 1 <= k <= D:
        raise ValueError("ref must be a 1-based part index")
    dd = int(ddof)
    if dd not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    if n - dd <= 0:
        raise ValueError("not enough compositions for this ddof")
    idx = [i for i in range(1, D + 1) if i != k]
    P = [[v / sum(row) for v in row] for row in Xm]
    Y = [[math.log(r[i - 1]) - math.log(r[k - 1]) for i in idx] for r in P]
    p = D - 1
    mu = [sum(Y[r][j] for r in range(n)) / n for j in range(p)]
    Sg = [[sum((Y[r][a] - mu[a]) * (Y[r][b] - mu[b]) for r in range(n)) / (n - dd)
           for b in range(p)] for a in range(p)]
    L = C.chol(Sg)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(p))
    ll = 0.0
    for r in range(n):
        y = [Y[r][j] - mu[j] for j in range(p)]
        w = C.solvev(Sg, y)
        q = sum(a * b for a, b in zip(y, w))
        ll += (-0.5 * p * math.log(2.0 * math.pi) - 0.5 * logdet
               - sum(math.log(v) for v in P[r]) - 0.5 * q)
    return RichResult(payload={
        "mu": mu, "Sigma": Sg, "ref": k, "loglik": ll, "n": n, "D": D,
        "method": "Additive logistic normal maximum likelihood fit"})


logistic_normal_fit = lognormfit


def cheatsheet():
    return 'aitlnf: Maximum likelihood fit of the additive logistic normal distribution.'
