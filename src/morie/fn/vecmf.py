# morie.fn -- function file (rootcoder007/morie)
"""VECM estimation with error-correction (Johansen 1995)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vecm"]


def vecm(Y, k_ar=1, coint_rank=1):
    r"""Estimate a Vector Error-Correction Model.

    .. math::

        \Delta Y_t = \alpha\beta' Y_{t-1}
                     + \sum_{i=1}^{p-1}\Gamma_i \Delta Y_{t-i} + \epsilon_t.

    Parameters
    ----------
    Y : array-like, shape (T, k)
        Panel of I(1) variables.
    k_ar : int, default 1
        Number of lagged differences (Γ-terms) in the VECM.
    coint_rank : int, default 1
        Cointegration rank ``r`` (number of stationary linear
        combinations).

    Returns
    -------
    RichResult
        keys: ``alpha`` (loading matrix, k × r), ``beta`` (cointegration
        vectors, k × r), ``Gamma`` (list of k × k lag matrices), ``Sigma``
        (residual covariance), ``loglik``, ``n``, ``k``, ``rank``,
        ``method``.

    References
    ----------
    Johansen S (1995). *Likelihood-Based Inference in Cointegrated Vector
    Autoregressive Models*. Oxford UP.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    T, k = Y.shape
    if T < 20 or k < 2 or coint_rank < 1 or coint_rank > k:
        raise ValueError(f"Need T>=20, k>=2, 1<=rank<=k; got T={T}, k={k}, r={coint_rank}.")

    from ._ts_core import VECM

    fit = VECM(Y, k_ar_diff=k_ar, coint_rank=coint_rank).fit()
    return RichResult(
        payload={
            "alpha": np.asarray(fit.alpha),
            "beta": np.asarray(fit.beta),
            "Gamma": ([np.asarray(g) for g in fit.gamma.reshape(k_ar, k, k)]
                      if k_ar > 0 else []),
            "Sigma": np.asarray(fit.sigma_u),
            "loglik": float(fit.llf),
            "n": int(T),
            "k": int(k),
            "rank": int(coint_rank),
            "method": "VECM by Johansen reduced-rank regression, beta "
                      "normalised to an identity leading block",
        }
    )


def cheatsheet():
    return "vecmf: VECM estimation (Johansen 1995)."
