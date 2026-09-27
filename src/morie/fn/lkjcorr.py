"""LKJ distribution on correlation matrices.

Lewandowski, D., Kurowicka, D. & Joe, H. (2009). Generating random correlation matrices based on vines and
extended onion method. Journal of Multivariate Analysis 100, 1989-2001.
"""

import math

from ._mvcore import as_matrix, cholesky, logdet_chol
from ._richresult import RichResult

__all__ = ["lkjcorr"]


def lkjcorr(R, eta=1.0):
    r"""f(R) = det(R)^(eta - 1) / c_d(eta), with

    c_d(eta) = 2^{sum_{k=1}^{d-1} (2 eta - 2 + d - k)(d - k)} prod_{k=1}^{d-1} B(eta + (d - k - 1)/2, eta + (d - k - 1)/2)^{d - k}.

    For d = 2 the correlation r has density (1 - r^2)^(eta - 1) / (2^{2 eta - 1} B(eta, eta)).

    Parameters
    ----------
    R : d x d correlation matrix (unit diagonal, positive definite)
    eta : float
        Shape, > 0 (eta = 1 is uniform over correlation matrices).

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, log_normalizer.

    References
    ----------
    Lewandowski, Kurowicka & Joe (2009). JMVA 100, 1989-2001, Sec 3.2.

    Examples
    --------
    >>> round(lkjcorr([[1.0, 0.0], [0.0, 1.0]], eta=1.0)["pdf"], 12)
    0.5
    """
    A = as_matrix(R)
    d = len(A)
    if not eta > 0 or any(abs(A[i][i] - 1) > 1e-12 for i in range(d)):
        raise ValueError("R must be a correlation matrix and eta > 0")
    ld = logdet_chol(cholesky(A))
    logc = 0.0
    for k in range(1, d):
        b = eta + (d - k - 1) / 2
        logc += (2 * eta - 2 + d - k) * (d - k) * math.log(2) + (d - k) * (2 * math.lgamma(b) - math.lgamma(2 * b))
    lp = (eta - 1) * ld - logc
    return RichResult(
        title="LKJ correlation",
        summary_lines=[("logpdf", lp)],
        payload={"pdf": math.exp(lp), "logpdf": lp, "log_normalizer": logc},
    )


def cheatsheet():
    return "lkjcorr: LKJ density det(R)^(eta-1)/c_d(eta) on correlation matrices."
