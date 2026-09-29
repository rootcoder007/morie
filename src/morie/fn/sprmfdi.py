# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit direct/indirect MEs."""

from ._richresult import RichResult
from .spprmf import _impacts


def sprmfdi(coef, rho, X, W, method="lesage_pace"):
    r"""Average direct, indirect and total marginal effects of a SAR probit model.

    The summary impacts of :func:`morie.fn.spprmf.spprmf` (LeSage and Pace
    2009, sec. 10.1.6): direct = mean of diag(phi(eta)) S^{-1} times
    beta_r, total = mean row sum, indirect = total - direct; returns
    only the three averaged vectors.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> r = sprmfdi([-0.2, 0.9], 0.4, X, W)
    >>> round(r["indirect"][0], 10)
    0.1526759112
    """
    r = _impacts(coef, rho, X, W, "probit", method)
    return RichResult(payload={"direct": r["direct"], "indirect": r["indirect"], "total": r["total"]})


sprmfdi_fn = sprmfdi


def cheatsheet() -> str:
    return "sprmfdi(coef, rho, X, W) -> SAR probit average direct, indirect and total effects."
