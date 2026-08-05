# morie.fn -- function file (rootcoder007/morie)
"""Closed-form 2-Wasserstein distance between two Gaussians."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_wasserstein_gauss"]


def ot_wasserstein_gauss(mu1, Sigma1, mu2, Sigma2):
    """Wasserstein distance between Gaussians, without solving anything.

    The Gaussian family is closed under optimal transport: the optimal map
    is affine and the cost splits into a mean part and a covariance part,
    the latter being the Bures metric.  This is the one multivariate case
    with a closed form, which makes it the natural anchor for every
    numerical transport solver.

    Formula: ``W_2^2 = ||m1 - m2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2
    S1^{1/2})^{1/2})`` -- Peyre & Cuturi (2019) eq. (2.41)-(2.42), p. 34,
    read from the rendered page; Olkin & Pukelsheim (1982).

    Parameters
    ----------
    mu1, mu2 : array-like, shape (d,)
        Mean vectors.
    Sigma1, Sigma2 : array-like, shape (d, d)
        Covariance matrices, symmetric positive semi-definite.

    Returns
    -------
    RichResult
        ``W2``, ``W2_sq``, ``mean_part``, ``bures_sq``, ``d``.

    References
    ----------
    Olkin, I. and Pukelsheim, F. (1982).  The distance between two random
    vectors with given dispersion matrices.  Linear Algebra and its
    Applications 48:257-263.  doi:10.1016/0024-3795(82)90112-4.
    """
    a = [float(t) for t in core.vec(mu1)]
    b = [float(t) for t in core.vec(mu2)]
    w2sq = ot.w2gauss(mu1, Sigma1, mu2, Sigma2)
    mp = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    return RichResult(payload={
        "W2": w2sq ** 0.5, "W2_sq": w2sq, "mean_part": mp,
        "bures_sq": w2sq - mp, "d": len(a),
        "method": "Gaussian 2-Wasserstein distance (Bures)"})


def cheatsheet():
    return "otwsg: closed-form 2-Wasserstein distance between two Gaussians"
