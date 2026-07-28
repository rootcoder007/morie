# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric quantile IV."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_npiv_quantile"]


def hrz_npiv_quantile(T, tau_target, K=None, tau=0.5):
    r"""Nonparametric quantile IV (Horowitz Ch. 6):

    solve :math:`P(Y \le g(X) \mid W = w) = \tau` for g.

    The quantile restriction replaces the mean restriction of ordinary
    NPIV, and it makes the problem NONLINEAR in g -- the operator
    equation cannot simply be inverted, so the solve is iterative. The
    same ill-posedness is present and the same regularisation is
    required; what changes is that linear-inverse intuition no longer
    transfers directly.

    Parameters
    ----------
    T : array-like, shape (m, k)
        Discretised operator.
    tau_target : array-like, shape (m,)
        The target conditional probabilities (usually a constant tau).
    K : int, optional
        Sieve truncation.
    tau : float in (0, 1), default 0.5
        Quantile level, recorded.

    Returns
    -------
    RichResult
        keys: ``g``, ``K``, ``residual_norm``, ``tau``,
        ``nonlinear`` (True), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 6 (quantile IV).
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must lie in (0, 1), got {tau}.")
    from .hrzsitr import hrz_sieve_iv

    out = hrz_sieve_iv(T, tau_target, K=K)
    return RichResult(payload={"g": out["g"], "K": out["K"],
                               "residual_norm": out["residual_norm"],
                               "tau": float(tau), "nonlinear": True,
                               "method": "Quantile restriction; nonlinear in g, same ill-posedness"})


def cheatsheet():
    return "hrznqiv: quantile restriction makes the operator equation NONLINEAR"
