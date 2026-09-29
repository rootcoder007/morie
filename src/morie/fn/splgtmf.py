# morie.fn -- function file (rootcoder007/morie)
"""Spatial logit marginal effects."""

from .spprmf import _impacts


def splgtmf(coef, rho, X, W, method="lesage_pace"):
    r"""Marginal effects (impacts) of the SAR logit model y* = rho W y* + X beta + e, logistic e.

    As :func:`morie.fn.spprmf.spprmf` with the logistic density
    Lambda(eta)(1 - Lambda(eta)) in place of phi: the effect matrix of
    covariate r is diag(lambda(eta)) S^{-1} beta_r with eta = S^{-1}
    X beta (LeSage and Pace 2009, sec. 10.1; Klier and McMillen 2008), and
    the average direct, indirect and total impacts are its mean diagonal,
    mean off-diagonal row sum and mean row sum.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.
    Klier, T. and McMillen, D. P. (2008). Clustering of auto supplier plants
    in the United States: generalized method of moments spatial logit for
    large samples. *JBES* 26, 460-471.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> r = splgtmf([-0.2, 0.9], 0.4, X, W)
    >>> round(r["total"][0], 10)
    0.3235962568
    """
    return _impacts(coef, rho, X, W, "logit", method)


splgtmf_fn = splgtmf


def cheatsheet() -> str:
    return "splgtmf(coef, rho, X, W) -> SAR logit average direct/indirect/total impacts."
