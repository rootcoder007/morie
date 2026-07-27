# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural model fit by inverse-probability-of-treatment weighting."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["marginal_structural_model"]


def marginal_structural_model(y, treatment_history, covariate_history, time=None):
    r"""Fit the MSM :math:`E[Y(\bar a)] = \beta_0 + \beta_1 \sum_t a_t` by IPTW.

    Computes stabilised inverse-probability-of-treatment weights

    .. math:: sw_i = \prod_t
              \frac{f(A_{it} \mid \bar A_{i,t-1})}
                   {f(A_{it} \mid \bar A_{i,t-1}, L_{it})},

    with both densities modelled by logistic regression, then fits the
    marginal structural model by weighted least squares of ``y`` on
    cumulative treatment. Under sequential exchangeability given the
    measured time-varying confounders :math:`L_t`, the weighted
    coefficient consistently estimates the causal per-period effect --
    which naive covariate adjustment cannot do when :math:`L_t` is
    itself affected by earlier treatment.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    treatment_history : array-like of {0, 1}, shape (n, T) or (n,)
        Treatment at each interval.
    covariate_history : array-like, shape (n, T) or (n,)
        Time-varying confounder measured at the start of each interval.
    time : ignored
        Accepted for backward compatibility with the placeholder
        signature.

    Returns
    -------
    RichResult
        keys: ``estimate`` (per-period causal effect beta_1),
        ``intercept``, ``weights`` (stabilised, n,), ``ess``, ``n``,
        ``n_periods``, ``method``.

    References
    ----------
    Robins, J. M., Hernan, M. A. & Brumback, B. (2000). Marginal
    structural models and causal inference in epidemiology.
    *Epidemiology*, 11(5), 550-560.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment_history, dtype=float)
    L = np.asarray(covariate_history, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if L.ndim == 1:
        L = L[:, None]
    n, T = A.shape
    if y.size != n or L.shape != (n, T):
        raise ValueError(f"shapes disagree: y {y.size}, A {A.shape}, L {L.shape}.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("treatment_history must be binary 0/1.")

    sw = np.ones(n)
    for t in range(T):
        past = A[:, :t]  # treatment history before t
        a_t = A[:, t]
        if a_t.min() == a_t.max():
            continue  # no variation: both densities are the indicator, ratio 1
        X_num = past if t > 0 else np.zeros((n, 0))
        X_den = np.column_stack([past, L[:, : t + 1]])
        p_num = _logit_fit(X_num, a_t) if X_num.shape[1] else np.full(n, a_t.mean())
        p_den = np.clip(_logit_fit(X_den, a_t), 1e-6, 1 - 1e-6)
        num = np.where(a_t == 1, p_num, 1 - p_num)
        den = np.where(a_t == 1, p_den, 1 - p_den)
        sw *= num / den

    cumA = A.sum(axis=1)
    D = np.column_stack([np.ones(n), cumA])
    W = sw[:, None]
    beta, *_ = np.linalg.lstsq(D * np.sqrt(W), y * np.sqrt(sw), rcond=None)
    ess = float(sw.sum() ** 2 / (sw**2).sum())

    return RichResult(
        payload={
            "estimate": float(beta[1]),
            "intercept": float(beta[0]),
            "weights": sw,
            "ess": ess,
            "n": int(n),
            "n_periods": int(T),
            "method": "Marginal structural model fit by IPTW (stabilised weights)",
        }
    )


def cheatsheet():
    return "msmest: MSM E[Y(abar)] = b0 + b1*sum(a_t) by stabilised IPTW WLS"
