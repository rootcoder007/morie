# morie.fn -- function file (rootcoder007/morie)
"""TMLE under time-varying confounding (sequential targeting)."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["tmle_time_varying_confound"]


def _expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def _logit(p):
    return np.log(p / (1 - p))


def tmle_time_varying_confound(y, A, L, regime=1.0, trunc=0.01):
    r"""Sequentially targeted g-computation for a static regime.

    Runs the iterated conditional expectation backwards, targeting at
    each step: starting from :math:`\bar Q_{T+1} = Y`, for t = T..1

    1. regress :math:`\bar Q_{t+1}` on the history
       :math:`(\bar A_t, \bar L_t)`;
    2. fluctuate that fit along the cumulative clever covariate
       :math:`H_t = \prod_{s \le t}
       \mathbb{1}\{A_s = a_s\}/g_s`;
    3. evaluate the targeted fit with :math:`A_t` set to the regime.

    Targeting at *every* time point -- not only the last -- is what
    makes the longitudinal estimator solve the full efficient
    influence-function equation; a single terminal fluctuation does
    not.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    A : array-like of {0, 1}, shape (n, T) or (n,)
        Treatment per period.
    L : array-like, shape (n, T) or (n,)
        Time-varying confounder.
    regime : scalar or array-like shape (T,), default 1
        Static regime.
    trunc : float, default 0.01
        Treatment-probability truncation.

    Returns
    -------
    RichResult
        keys: ``estimate`` (E[Y(abar)]), ``epsilons`` (per period),
        ``weights`` (final cumulative clever covariate), ``regime``,
        ``n_periods``, ``n``, ``method``.

    References
    ----------
    van der Laan, M. J. & Gruber, S. (2012). Targeted minimum
    loss-based estimation of causal effects of multiple time point
    interventions. *The International Journal of Biostatistics*, 8(1),
    Article 9.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if L.ndim == 1:
        L = L[:, None]
    n, T = A.shape
    if y.size != n or L.shape != (n, T):
        raise ValueError(f"shapes disagree: y {y.size}, A {A.shape}, L {L.shape}.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    reg = np.broadcast_to(np.asarray(regime, dtype=float).ravel(), (T,)).copy()
    trunc = float(trunc)
    if not 0 < trunc < 0.5:
        raise ValueError(f"trunc must lie in (0, 0.5), got {trunc}.")

    # treatment probabilities given the past
    gs = np.empty((n, T))
    for t in range(T):
        past = np.column_stack([A[:, :t], L[:, : t + 1]])
        if A[:, t].min() == A[:, t].max():
            gs[:, t] = np.clip(A[:, t].mean() if A[:, t].mean() > 0 else 1.0, trunc, 1 - trunc)
        else:
            gs[:, t] = np.clip(_logit_fit(past, A[:, t]), trunc, 1 - trunc)
    p_follow = np.where(A == reg[None, :], gs, 1 - gs)
    cum_g = np.cumprod(p_follow, axis=1)
    follows = np.cumprod((A == reg[None, :]).astype(float), axis=1)

    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    if span <= 0:
        raise ValueError("outcome has zero range.")
    Q = np.clip((y - lo) / span, 1e-6, 1 - 1e-6)

    eps_all = []
    for t in range(T - 1, -1, -1):
        X = np.column_stack([np.ones(n), A[:, : t + 1], L[:, : t + 1]])
        b, *_ = np.linalg.lstsq(X, Q, rcond=None)
        q_obs = np.clip(X @ b, 1e-6, 1 - 1e-6)
        H = follows[:, t] / cum_g[:, t]
        num = float(np.sum(H * (Q - q_obs)))
        den = float(np.sum(H**2 * q_obs * (1 - q_obs)))
        eps = num / den if den > 1e-14 else 0.0
        eps_all.append(float(eps))
        Xa = np.column_stack(
            [np.ones(n), A[:, :t], np.full(n, reg[t]), L[:, : t + 1]]
        )
        q_reg = np.clip(Xa @ b, 1e-6, 1 - 1e-6)
        Q = _expit(_logit(q_reg) + eps * (1.0 / cum_g[:, t]))

    est = float(Q.mean() * span + lo)
    return RichResult(
        payload={
            "estimate": est,
            "epsilons": np.array(eps_all[::-1]),
            "weights": follows[:, -1] / cum_g[:, -1],
            "regime": reg,
            "n_periods": int(T),
            "n": int(n),
            "method": "Sequentially targeted g-computation under time-varying confounding",
        }
    )


def cheatsheet():
    return "tmltvc: backwards ICE with a fluctuation at every t (not only the last)"
