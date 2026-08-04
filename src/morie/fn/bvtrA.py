# morie.fn -- function file (rootcoder007/morie)
"""Bias-variance decomposition of the expected prediction error."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['biasvardec', 'bias_variance_tradeoff']


def biasvardec(F, f, sigma2):
    """Bias-variance decomposition of the expected prediction error.

    Formula: E(y - fhat)^2 = Var(e) + Bias[fhat]^2 + Var(fhat)

    Parameters
    ----------
    F : array-like, shape (R, n)
        One row per replicate fit, one column per evaluation point: the predictions fhat(x_j) from replicate r.
    f : array-like
        The true f(x_j) at the same n evaluation points.
    sigma2 : float
        Irreducible error Var(e); must be non-negative.

    Returns
    -------
    RichResult
        ``bias2``, ``variance``, ``irreducible``, ``total``, ``bias2_point``, ``variance_point``, ``R``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Sect. 4.2 p. 113, which reproduces the decomposition of Hastie, Tibshirani and Friedman (2008) p. 223: the expected prediction error under quadratic loss splits into Var(e), the squared bias of fhat and the variance of fhat.  Averaged over the evaluation points here.  Read from the chapter PDF, not recalled.
    """
    F = C.mat(F)
    f = C.vec(f)
    s2 = float(sigma2)
    R = len(F)
    if R == 0:
        raise ValueError("F must have at least one replicate row")
    n = len(F[0])
    if n != len(f):
        raise ValueError("F must have one column per entry of f")
    if s2 < 0.0:
        raise ValueError("sigma2 must be non-negative")
    b2, vv = [], []
    for j in range(n):
        col = [F[r][j] for r in range(R)]
        m = sum(col) / R
        b2.append((m - f[j]) ** 2)
        vv.append(sum((v - m) ** 2 for v in col) / R)
    bias2 = sum(b2) / n
    var = sum(vv) / n
    return RichResult(payload={
        "bias2": bias2, "variance": var, "irreducible": s2,
        "total": s2 + bias2 + var, "bias2_point": b2, "variance_point": vv,
        "R": R, "n": n,
        "method": "Bias-variance decomposition, MVSML Sect. 4.2"})


bias_variance_tradeoff = biasvardec


def cheatsheet():
    return 'bvtrA: Bias-variance decomposition of the expected prediction error.'
