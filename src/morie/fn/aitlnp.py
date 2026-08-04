# morie.fn -- function file (rootcoder007/morie)
"""Density of the additive logistic-normal on the simplex."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lgtnpdf", "logistic_normal_pdf"]


def lgtnpdf(x, mu, Sigma):
    """Additive logistic-normal density at a composition.

    The factor that is easy to drop and fatal to drop is the Jacobian
    (prod_i x_i)^-1, over ALL D parts including the reference one.
    Without it the "density" does not integrate to one over the
    simplex and every likelihood built on it is wrong by a
    data-dependent factor -- which does not cancel between
    compositions, so even likelihood RATIOS come out wrong.

    Formula: f(x) = (2 pi)^{-(D-1)/2} |Sigma|^{-1/2}
                    (prod_{i=1}^{D} x_i)^{-1}
                    exp( -1/2 (alr(x) - mu)' Sigma^{-1} (alr(x) - mu) )

    Parameters
    ----------
    x : array-like
        A composition with D strictly positive parts.
    mu : array-like
        Mean of the alr coordinates, length D-1.
    Sigma : array-like, shape (D-1, D-1)
        Covariance of the alr coordinates, positive definite.

    Returns
    -------
    RichResult
        ``density``, ``log_density``, ``alr``, ``quadratic_form``,
        ``log_jacobian``, ``log_det``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 6, which defines the additive logistic-normal class as the
    law of alr^-1(Y) for Y normal, with the density carrying the
    Jacobian (prod x_i)^-1 of the additive log-ratio transform.  The
    reference part is the LAST, matching the sibling module ``aitalr``.
    """
    x = C.vec(x)
    D = len(x)
    if D < 2:
        raise ValueError("a composition needs at least two parts")
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    mu = C.vec(mu)
    if len(mu) != D - 1:
        raise ValueError("mu must have D-1 entries")
    S = C.mat(Sigma)
    if len(S) != D - 1 or any(len(r) != D - 1 for r in S):
        raise ValueError("Sigma must be (D-1) x (D-1)")
    L = C.chol(S)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(D - 1))
    y = [math.log(x[i]) - math.log(x[D - 1]) for i in range(D - 1)]
    dv = [y[i] - mu[i] for i in range(D - 1)]
    z = C.solvev(S, dv)
    q = sum(dv[i] * z[i] for i in range(D - 1))
    lj = -sum(math.log(v) for v in x)
    ld = (-0.5 * (D - 1) * math.log(2.0 * math.pi) - 0.5 * logdet
          + lj - 0.5 * q)
    return RichResult(payload={
        "density": math.exp(ld), "log_density": ld, "alr": y,
        "quadratic_form": q, "log_jacobian": lj, "log_det": logdet,
        "D": float(D),
        "method": "Additive logistic-normal density, Aitchison Chapter 6"})


logistic_normal_pdf = lgtnpdf


def cheatsheet():
    return "aitlnp: normal on alr coords times the Jacobian (prod x_i)^-1"
