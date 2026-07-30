# morie.fn -- function file (rootcoder007/morie)
"""Gaussian mixture density -- ESL Sec 6.8."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .eslemg import esl_em_gmm, _log_mvn

__all__ = ["esl_gaussian_mixture"]


def esl_gaussian_mixture(X, k=2, newdata=None, **kwargs):
    r"""Fit a Gaussian mixture and evaluate its density.

    .. math::
        f(x) = \sum_{j=1}^{k} \pi_j\, \mathcal{N}(x \mid \mu_j, \Sigma_j).

    Fitting is delegated to :func:`~morie.fn.eslemg.esl_em_gmm`; what this
    adds is the fitted density itself, evaluated at the training points or at
    ``newdata``. ESL Sec 6.8 presents the mixture as a *density estimate* --
    a smoother with an adaptive, data-driven bandwidth -- rather than as a
    clustering device, and the density is the object that view needs.

    Parameters
    ----------
    X : array-like
        Training data, shape ``(n, p)``.
    k : int
        Number of components.
    newdata : array-like, optional
        Points at which to evaluate the density. Defaults to ``X``.
    **kwargs
        Passed through to :func:`~morie.fn.eslemg.esl_em_gmm`.

    Returns
    -------
    RichResult
        ``density`` and ``log_density`` at the evaluation points, plus the
        fitted ``pi``, ``mu``, ``sigma``, ``loglik``, ``aic``, ``bic``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    The fitted density integrates to one, which is the check that separates a
    density estimate from an arbitrary positive function.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-3, 1, 400), rng.normal(3, 1, 400)]
    >>> grid = np.linspace(-12, 12, 2001)
    >>> d = esl_gaussian_mixture(X, k=2, newdata=grid, seed=1)["density"]
    >>> bool(abs(np.trapezoid(d, grid) - 1.0) < 1e-3)
    True

    Density is higher at a mode than in the valley between the components.

    >>> r = esl_gaussian_mixture(X, k=2, newdata=[-3.0, 0.0], seed=1)
    >>> bool(r["density"][0] > r["density"][1])
    True
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    fit = esl_em_gmm(X, k=k, **kwargs)
    Z = X if newdata is None else np.asarray(newdata, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[1] != X.shape[1]:
        raise ValueError(f"newdata has {Z.shape[1]} columns but X has {X.shape[1]}")
    comp = np.empty((Z.shape[0], k))
    for j in range(k):
        comp[:, j] = np.log(fit["pi"][j] + 1e-300) + _log_mvn(Z, fit["mu"][j], fit["sigma"][j])
    mx = comp.max(axis=1, keepdims=True)
    logd = mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))
    return RichResult(
        title="Gaussian mixture density",
        summary_lines=[("k", int(k)), ("eval points", int(Z.shape[0]))],
        payload={
            "density": np.exp(logd), "log_density": logd,
            "pi": fit["pi"], "mu": fit["mu"], "sigma": fit["sigma"],
            "loglik": fit["loglik"], "aic": fit["aic"], "bic": fit["bic"],
            "resp": fit["resp"], "labels": fit["labels"],
            "method": "esl_gaussian_mixture",
        },
    )


def cheatsheet():
    return "eslmix: mixture as a DENSITY estimate (ESL 6.8); integrates to 1, unlike a bare cluster fit"
