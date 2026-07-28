# morie.fn -- function file (rootcoder007/morie)
"""Quadratic expansion of the log profile likelihood."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch3_log_profile_expansion"]


def kosorok_ch3_log_profile_expansion(theta_bar_n, theta_hat_n, I_tilde,
                                      log_pl_hat=0.0, n=None):
    r"""Quadratic expansion of the log profile likelihood (Kosorok
    Ch. 3):

    .. math:: \log pl_n(\bar\theta_n) = \log pl_n(\hat\theta_n)
              - \tfrac12 n (\bar\theta_n - \hat\theta_n)'
              \tilde I (\bar\theta_n - \hat\theta_n) + o_P(1).

    The profile likelihood behaves like an ordinary parametric
    likelihood with the EFFICIENT information in the quadratic term,
    which is what licenses profile-likelihood confidence intervals in
    a semiparametric model. The remainder is o_P(1) and is not
    modelled here; the returned value is the leading expansion, and
    the docstring says so rather than implying exactness.

    Parameters
    ----------
    theta_bar_n, theta_hat_n : array-like
        The evaluation point and the maximiser.
    I_tilde : array-like
        Efficient information matrix (or scalar).
    log_pl_hat : float, default 0.0
        Log profile likelihood at the maximiser.
    n : int, optional
        Sample size; required for the n scaling.

    Returns
    -------
    RichResult
        keys: ``log_pl``, ``quadratic_term``, ``delta``,
        ``lrt_statistic`` (2 x the drop, chi-square calibrated),
        ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3 (log profile likelihood expansion).
    """
    tb = np.atleast_1d(np.asarray(theta_bar_n, dtype=float))
    th = np.atleast_1d(np.asarray(theta_hat_n, dtype=float))
    if tb.shape != th.shape:
        raise ValueError("theta_bar_n and theta_hat_n must have the same shape.")
    I = np.atleast_2d(np.asarray(I_tilde, dtype=float))
    if I.shape[0] != tb.size or I.shape[1] != tb.size:
        raise ValueError(f"I_tilde must be {tb.size}x{tb.size}.")
    if n is None:
        raise ValueError("n is required for the n-scaling of the quadratic term.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    d = tb - th
    quad = 0.5 * n * float(d @ I @ d)
    return RichResult(
        payload={"log_pl": float(log_pl_hat) - quad, "quadratic_term": quad,
                 "delta": d, "lrt_statistic": 2.0 * quad, "n": n,
                 "method": "log pl(theta_bar) = log pl(theta_hat) - n d'I d/2 + o_P(1)"}
    )


def cheatsheet():
    return "ksr071: profile likelihood is quadratic with the EFFICIENT information"
