# morie.fn -- function file (rootcoder007/morie)
"""Propensity-only TMLE -- consistent when the outcome model is misspecified."""

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate

__all__ = ["tmle_propensity_only"]


def tmle_propensity_only(y, D, X, trunc=0.01):
    r"""TMLE with a deliberately null initial outcome fit.

    Setting :math:`\bar Q^0` to a constant (the arm-free sample mean)
    makes the targeting step carry all the information: the
    fluctuation along :math:`H = A/g - (1-A)/(1-g)` recovers a
    weighting estimator. This is the concrete demonstration of double
    robustness -- the outcome model is as wrong as it can be, yet the
    estimate stays consistent provided the propensity model is right.
    The fully specified TMLE is returned alongside for comparison.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates for the propensity model.
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``ate``, ``se``, ``ci``, ``ate_full`` (TMLE with a real
        outcome model), ``epsilon``, ``n``, ``method``.

    References
    ----------
    van der Laan, M. J. & Rose, S. (2011). *Targeted Learning: Causal
    Inference for Observational and Experimental Data*. Springer.
    Ch. 4-5 (double robustness of TMLE).
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    # null outcome model: a single intercept column carries no covariate signal
    null_W = np.zeros((y.size, 1))
    from .aiptdd import _logit_fit

    g = _logit_fit(X, D)
    out = tmle_ate(y, D, null_W, trunc=trunc, g=g)
    full = tmle_ate(y, D, X, trunc=trunc)

    return RichResult(
        payload={
            "ate": out["ate"],
            "se": out["se"],
            "ci": out["ci"],
            "ate_full": full["ate"],
            "epsilon": out["epsilon"],
            "n": out["n"],
            "method": "Propensity-only TMLE (null initial Q; consistency rests on g)",
        }
    )


def cheatsheet():
    return "tmlpoy: constant initial Q, real g -- targeting alone recovers the ATE"
