# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit model (redirect to the real estimator)."""

from __future__ import annotations

from .sarbayes import sar_probit_gibbs
from .spdiscrete import spatial_probit_gmm


def spatial_probit(y, X, W, *, method: str = "gmm", **kwargs):
    r"""Spatial probit model.

    ``method="gmm"`` (default) is the GMM spatial autoregressive probit of
    Pinkse and Slade (1998) as implemented by McMillen
    (:func:`morie.fn.spdiscrete.spatial_probit_gmm`); ``method="bayes"`` is the
    Bayesian SAR probit Gibbs sampler of LeSage and Pace (2009, ch. 10)
    (:func:`morie.fn.sarbayes.sar_probit_gibbs`). ``kwargs`` go to the chosen
    estimator. This module formerly returned the variance of its input.

    Parameters
    ----------
    y : 0/1 outcomes.
    X : design matrix with an intercept column.
    W : spatial weights matrix (usually row-standardised).
    method : ``"gmm"`` or ``"bayes"``.

    Returns
    -------
    RichResult
        The result of the chosen estimator (``coefficients``, ``rho``, ...).

    References
    ----------
    Pinkse, J. and Slade, M. E. (1998). Contracting in space: an application of
    spatial statistics to discrete-choice models. *Journal of Econometrics* 85, 125-154.

    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, chapter 10.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    >>> spatial_probit(y, X, W).rho == spatial_probit_gmm(y, X, W).rho
    True
    >>> len(spatial_probit(y, X, W, method="bayes", ndraw=20, burn_in=5).rho_draws)
    20
    """
    if method == "gmm":
        return spatial_probit_gmm(y, X, W, **kwargs)
    if method == "bayes":
        return sar_probit_gibbs(y, X, W, **kwargs)
    raise ValueError("method must be 'gmm' or 'bayes'")


spat = spatial_probit


def cheatsheet() -> str:
    return "spatial_probit(y, X, W, ...) -> Spatial probit model"


# compact alias per ledger/NAMING.md
spatialprobit = spatial_probit
