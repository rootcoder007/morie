# morie.fn -- function file (rootcoder007/morie)
"""Minimum description length -- ESL Sec 7.8."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_mdl"]


def esl_mdl(loglik, theta, n=None, prior_sd=None):
    r"""Two-part MDL description length, with its BIC equivalence.

    The message length is the cost of transmitting the data given the model
    plus the cost of transmitting the model:

    .. math::
        \mathrm{MDL} = -\log P(y \mid X, \theta) - \log P(\theta).

    ESL Sec 7.8 shows that with the standard :math:`\tfrac{1}{2}\log n` bits
    per parameter this reduces to

    .. math::
        \mathrm{MDL} \;=\; -\ell + \tfrac{d}{2}\log n \;=\; \tfrac{1}{2}\,\mathrm{BIC},

    so minimising description length and minimising BIC pick the same model.
    Both are returned, and their identity is asserted in the doctest, because
    the equivalence is the point of the section.

    Units: with a natural logarithm the result is in *nats*; ``bits`` divides
    by :math:`\log 2` for the description-length reading.

    Parameters
    ----------
    loglik : float
        Maximised log-likelihood :math:`\ell`.
    theta : array-like or int
        Fitted parameters, or just their count.
    n : int, optional
        Sample size. Required for the BIC-equivalent parameter cost; without
        it a Gaussian prior cost is used instead and ``bic`` is NaN.
    prior_sd : float, optional
        If given (with ``theta`` as values), the parameter cost is the
        negative log density of a ``N(0, prior_sd^2)`` prior instead of
        :math:`\tfrac{d}{2}\log n`.

    Returns
    -------
    RichResult
        ``mdl`` (nats), ``bits``, ``data_cost``, ``model_cost``, ``bic``,
        ``aic``, ``d``.

    References
    ----------
    Rissanen, J. (1978). Modeling by shortest data description.
        *Automatica*, 14(5), 465-471.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    MDL is exactly half of BIC under the standard parameter cost.

    >>> r = esl_mdl(loglik=-120.0, theta=4, n=100)
    >>> bool(abs(r["mdl"] - r["bic"] / 2) < 1e-12)
    True
    >>> float(round(r["model_cost"], 6))
    9.21034

    More parameters at the same fit cost more to describe, which is the
    whole mechanism.

    >>> bool(esl_mdl(-120.0, 8, n=100)["mdl"] > esl_mdl(-120.0, 4, n=100)["mdl"])
    True

    >>> esl_mdl(-10.0, 2)["bic"]
    nan
    """
    d = int(theta) if np.isscalar(theta) else int(np.asarray(theta).size)
    if d < 0:
        raise ValueError("the parameter count must be non-negative")
    ll = float(loglik)

    if prior_sd is not None:
        if np.isscalar(theta):
            raise ValueError("prior_sd needs theta to be the parameter values, not a count")
        if prior_sd <= 0:
            raise ValueError("prior_sd must be positive")
        th = np.asarray(theta, dtype=float).ravel()
        model_cost = float(
            0.5 * np.sum(th**2) / prior_sd**2 + d * np.log(prior_sd * np.sqrt(2 * np.pi))
        )
    elif n is not None:
        if n < 1:
            raise ValueError("n must be at least 1")
        model_cost = float(0.5 * d * np.log(n))
    else:
        model_cost = float(0.5 * d * np.log(2 * np.pi))

    mdl = -ll + model_cost
    bic = float(d * np.log(n) - 2 * ll) if (n is not None and prior_sd is None) else float("nan")
    return RichResult(
        title="Minimum description length",
        summary_lines=[("d", d), ("MDL (nats)", mdl), ("BIC", bic)],
        payload={
            "mdl": mdl, "bits": mdl / np.log(2),
            "data_cost": -ll, "model_cost": model_cost,
            "bic": bic, "aic": float(2 * d - 2 * ll),
            "d": d, "loglik": ll,
            "method": "esl_mdl",
        },
    )


def cheatsheet():
    return "eslmdl: MDL = -loglik + (d/2)log n = BIC/2 exactly; `bits` divides by log 2"
