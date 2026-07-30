# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric quantile IV."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_npiv_quantile", "horowitz_nonpar_quantile_iv"]


def hrz_npiv_quantile(T, tau_target, K=None, tau=0.5):
    r"""Nonparametric quantile IV (Horowitz Sec. 5.3-5.5):

    solve :math:`P(Y \le g(X) \mid W = w) = \tau` for g.

    The quantile restriction replaces the mean restriction of ordinary
    NPIV, and in full generality it makes the problem NONLINEAR in g.
    THIS implementation does NOT perform that nonlinear solve: it runs
    the linear sieve-truncation solve of the mean restriction and
    records tau alongside it, so the quantile restriction never enters
    the arithmetic. Treat the output as the mean-NPIV solution with a
    quantile label, not as a quantile-IV estimate. The
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
        ``nonlinear`` (False -- see above), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 5, Sec. 5.5.1 (nonparametric quantile IV).
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must lie in (0, 1), got {tau}.")
    from .hrzsitr import hrz_sieve_iv

    out = hrz_sieve_iv(T, tau_target, K=K)
    return RichResult(payload={"g": out["g"], "K": out["K"],
                               "residual_norm": out["residual_norm"],
                               "tau": float(tau), "nonlinear": False,
                               "method": "Linear sieve solve of the MEAN restriction; tau recorded, not enforced"})


def cheatsheet():
    return "hrznqiv: quantile restriction makes the operator equation NONLINEAR"


#: Catalogue alias for :func:`hrz_npiv_quantile`.
horowitz_nonpar_quantile_iv = hrz_npiv_quantile
