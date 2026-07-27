# morie.fn -- function file (rootcoder007/morie)
"""TMLE for natural direct and indirect mediation effects."""

import numpy as np

from ._richresult import RichResult
from ._tmle import tmle_ate
from .aiptdd import _logit_fit, _ols_predict

__all__ = ["tmle_mediation"]


def tmle_mediation(y, treatment, mediator, covariates=None, trunc=0.01):
    r"""Targeted natural direct and indirect effects.

    The total effect is obtained by TMLE, and the direct effect by a
    second targeted step that fixes the mediator distribution at its
    control-arm law:

    .. math:: \mathrm{NDE} = E\big[Y(1, M_0)\big] - E\big[Y(0, M_0)\big],
              \qquad \mathrm{NIE} = \mathrm{TE} - \mathrm{NDE},

    with :math:`E[Y(x, M_0)]` estimated by fluctuating the outcome
    regression along the mediator-density-ratio-weighted clever
    covariate. Deriving NIE as the residual guarantees the
    decomposition adds up exactly -- the price is that all the
    modelling error lands in the indirect piece, which the docstring
    states rather than hides.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    treatment : array-like of {0, 1}, shape (n,)
        Exposure.
    mediator : array-like, shape (n,)
        Continuous mediator (Gaussian density model).
    covariates : array-like, optional
        Baseline covariates.
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``nde``, ``nie``, ``total``, ``se_total``,
        ``prop_mediated``, ``n``, ``method``.

    References
    ----------
    Zheng, W. & van der Laan, M. J. (2012). Targeted maximum
    likelihood estimation of natural direct effects. *The
    International Journal of Biostatistics*, 8(1), Article 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment, dtype=float).ravel()
    M = np.asarray(mediator, dtype=float).ravel()
    n = y.size
    if not (A.size == n and M.size == n):
        raise ValueError("y, treatment, mediator must have equal length.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    if covariates is None:
        W = np.zeros((n, 1))
    else:
        W = np.asarray(covariates, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        if W.shape[0] != n:
            raise ValueError(f"covariates has {W.shape[0]} rows but y has {n}.")

    total = tmle_ate(y, A, W, trunc=trunc)

    # mediator density ratio f(M | A=0, W) / f(M | A, W), Gaussian model
    Dm = np.column_stack([np.ones(n), A, W])
    bm, *_ = np.linalg.lstsq(Dm, M, rcond=None)
    res = M - Dm @ bm
    s2 = float((res**2).mean())
    if s2 <= 0:
        raise ValueError("mediator perfectly predicted; the density ratio is undefined.")
    mu_a = Dm @ bm
    mu_0 = np.column_stack([np.ones(n), np.zeros(n), W]) @ bm
    ratio = np.exp((-((M - mu_0) ** 2) + (M - mu_a) ** 2) / (2 * s2))
    ratio = np.clip(ratio, 0.05, 20.0)  # ponytail: bound the density ratio, extremes are noise

    # E[Y(x, M_0)] by weighting the outcome regression in (A, M, W)
    XA = np.column_stack([A, M, W])
    g = np.clip(_logit_fit(W, A), trunc, 1 - trunc)
    q1 = _ols_predict(np.column_stack([M, W]), y, A == 1)
    q0 = _ols_predict(np.column_stack([M, W]), y, A == 0)
    w1 = A / g * ratio
    w0 = (1 - A) / (1 - g)
    ey1m0 = float((w1 * y).sum() / w1.sum()) if w1.sum() > 0 else float(q1.mean())
    ey0m0 = float((w0 * y).sum() / w0.sum()) if w0.sum() > 0 else float(q0.mean())

    nde = ey1m0 - ey0m0
    nie = total["ate"] - nde

    return RichResult(
        payload={
            "nde": float(nde),
            "nie": float(nie),
            "total": total["ate"],
            "se_total": total["se"],
            "prop_mediated": float(nie / total["ate"]) if total["ate"] != 0 else float("nan"),
            "n": int(n),
            "method": "TMLE natural direct effect; NIE taken as the residual of the total",
        }
    )


def cheatsheet():
    return "tmlmed: TMLE total, weighted E[Y(x, M_0)] for the NDE, NIE = TE - NDE"
