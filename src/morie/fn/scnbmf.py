# morie.fn -- function file (rootcoder007/morie)
"""Spatial NB marginal effects: direct, indirect and total impacts of a spatial-lag NB2 model."""

from __future__ import annotations

from ._richresult import RichResult
from .scpmf import _count_impacts


def scnbmf(coef, rho, X, W) -> RichResult:
    r"""Marginal effects (impacts) of the spatial-lag negative binomial (NB2) model.

    The NB2 conditional mean ``exp((I - rho W)^{-1} X beta)`` is that of the
    Poisson model (the dispersion ``theta`` enters only the variance), so the
    impacts are those of :func:`morie.fn.scpmf.scpmf`: ``S_k = diag(mu) (I -
    rho W)^{-1} beta_k``, direct ``tr(S_k) / n``, total ``1' S_k 1 / n``,
    indirect their difference.

    Parameters
    ----------
    coef : coefficients ``beta``.
    rho : spatial lag parameter.
    X : ``n x p`` design matrix.
    W : ``n x n`` spatial weights matrix.

    Returns
    -------
    RichResult
        ``direct``, ``indirect``, ``total``, ``fitted``, ``linear_predictor``.

    References
    ----------
    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, section 2.7.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = scnbmf([0.2, 0.5], 0.3, [[1, 0.1], [1, 0.4], [1, 0.9]], W)
    >>> round(r.total[1], 10)
    1.337365737
    """
    return _count_impacts(coef, rho, X, W)


scnbmf_fn = scnbmf


def cheatsheet() -> str:
    return "scnbmf(coef, rho, X, W) -> direct/indirect/total impacts of the spatial-lag NB2 model."
