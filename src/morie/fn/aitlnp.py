# morie.fn -- function file (rootcoder007/morie)
"""Density of the additive logistic normal distribution on the simplex."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lognormpdf', 'logistic_normal_pdf']


def lognormpdf(x, mu, Sigma):
    """Density of the additive logistic normal distribution on the simplex.

    Formula: f(x) = (2 pi)^(-(D-1)/2) |Sigma|^(-1/2) (prod_i x_i)^-1 exp( -0.5 (alr(x) - mu)' Sigma^-1 (alr(x) - mu) )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts, length D.
    mu : array-like
        Mean of the additive log-ratio coordinates, length D - 1.
    Sigma : array-like, shape (D-1, D-1)
        Covariance of the additive log-ratio coordinates; must be positive definite.

    Returns
    -------
    RichResult
        ``density``, ``log_density``, ``alr``, ``quadratic_form``, ``log_jacobian``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The log-ratio algebra and the additive logistic normal law were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sects. 4.1 and 4.3, which attribute the law to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  A composition is additive logistic normal when its alr transform is multivariate normal.  Sect. 4.3 eq (15) prints the classical logistic normal density in ilr coordinates with Jacobian (sqrt(D) x_1 x_2 ... x_D)^-1.  In the alr coordinates used here the sqrt(D) contributed by the ilr basis is absent, giving (x_1 x_2 ... x_D)^-1.  That factor was re-derived rather than assumed: with y_i = log(x_i/x_D) and free coordinates x_1..x_{D-1}, dy/dx = diag(1/x_i) + (1/x_D) 1 1', whose determinant is (prod_{i<D} 1/x_i)(1 + (1 - x_D)/x_D) = 1 / prod_{i=1}^{D} x_i.  The composition is closed to sum 1 before the density is evaluated, because the density is with respect to the simplex of unit total; the log-ratios are unchanged by that closure, only the Jacobian factor depends on it.
    """
    x = C.vec(x)
    D = len(x)
    if D < 2:
        raise ValueError("the logistic normal needs at least two parts")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    s = sum(x)
    x = [v / s for v in x]
    mu = C.vec(mu)
    if len(mu) != D - 1:
        raise ValueError("mu must have D - 1 entries")
    Sg = C.mat(Sigma)
    if len(Sg) != D - 1 or len(Sg[0]) != D - 1:
        raise ValueError("Sigma must be (D - 1) by (D - 1)")
    L = C.chol(Sg)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(D - 1))
    lr = math.log(x[D - 1])
    y = [math.log(x[i]) - lr - mu[i] for i in range(D - 1)]
    w = C.solvev(Sg, y)
    q = sum(a * b for a, b in zip(y, w))
    lj = -sum(math.log(v) for v in x)
    ll = (-0.5 * (D - 1) * math.log(2.0 * math.pi) - 0.5 * logdet + lj - 0.5 * q)
    return RichResult(payload={
        "density": math.exp(ll), "log_density": ll,
        "alr": [math.log(x[i]) - lr for i in range(D - 1)],
        "quadratic_form": q, "log_jacobian": lj, "D": D,
        "method": "Additive logistic normal density"})


logistic_normal_pdf = lognormpdf


def cheatsheet():
    return 'aitlnp: Density of the additive logistic normal distribution on the simplex.'
