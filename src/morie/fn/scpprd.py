# morie.fn -- function file (rootcoder007/morie)
"""Spatial Poisson predicted counts: fitted means of a spatial-lag count model."""

from __future__ import annotations

from ._richresult import RichResult
from .scpmf import _count_impacts


def scpprd(coef, X, W, rho=0.2) -> RichResult:
    r"""Predicted counts ``mu = exp((I - rho W)^{-1} X beta)`` of the spatial-lag Poisson (or NB2) model.

    The spatial multiplier ``(I - rho W)^{-1}`` acts on the linear predictor
    (Lambert, Brown and Florax 2010), so each unit's expected count depends
    on the regressors of all units.

    Parameters
    ----------
    coef : coefficients ``beta``.
    X : ``n x p`` design matrix.
    W : ``n x n`` spatial weights matrix.
    rho : spatial lag parameter.

    Returns
    -------
    RichResult
        ``fitted`` (``mu``), ``linear_predictor`` and ``total`` (the sum of ``mu``).

    References
    ----------
    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step
    estimator for a spatial lag model of counts. *Regional Science and Urban
    Economics* 40, 241-252.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> [round(v, 10) for v in scpprd([0.2, 0.5], [[1, 0.1], [1, 0.4], [1, 0.9]], W, rho=0.3).fitted]
    [1.5316929491, 1.8002257756, 2.2850173707]
    """
    r = _count_impacts(coef, rho, X, W)
    mu = r.fitted
    s = 0.0
    for v in mu:
        s += v
    return RichResult(payload={"fitted": mu, "linear_predictor": r.linear_predictor, "total": s})


scpprd_fn = scpprd


def cheatsheet() -> str:
    return "scpprd(coef, X, W, rho) -> fitted means exp((I - rho W)^-1 X beta)."
