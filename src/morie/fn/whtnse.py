# morie.fn -- function file (rootcoder007/morie)
"""Hosking's multivariate portmanteau test for white noise."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["white_noise_test"]


def _autocov(E, lag):
    r"""\hat\Gamma_\ell = n^{-1} \sum_{t=\ell+1}^{n} e_t e_{t-\ell}'."""
    n = E.shape[0]
    if lag == 0:
        return (E.T @ E) / n
    return (E[lag:].T @ E[:-lag]) / n


def white_noise_test(X, lags=10, cdf=None, fitdf=0, modified=True):
    r"""Test a multivariate series for white noise (Hosking 1980).

    Hosking generalised the univariate portmanteau statistic to several
    series at once. His modified form is

    .. math::

        \tilde Q_m = n^2 \sum_{\ell=1}^{m} (n-\ell)^{-1}\,
                     \hat r_\ell'\,(\hat R_0^{-1} \otimes \hat R_0^{-1})\,
                     \hat r_\ell

    with :math:`\hat r_\ell = \mathrm{vec}(\hat R_\ell')`,
    :math:`\hat R_\ell = \hat L'\hat\Gamma_\ell\hat L`,
    :math:`\hat L\hat L' = \hat\Gamma_0^{-1}`, and
    :math:`\hat\Gamma_\ell = n^{-1}\sum_{t=\ell+1}^{n} e_t e_{t-\ell}'`.

    The Kronecker form collapses. Because :math:`\hat L\hat L' =
    \hat\Gamma_0^{-1}`, the standardised lag-zero matrix
    :math:`\hat R_0 = \hat L'\hat\Gamma_0\hat L` is the identity, so the
    quadratic form is just :math:`\mathrm{tr}(\hat R_\ell'\hat R_\ell)`.
    Substituting back gives the trace form computed here:

    .. math::

        \tilde Q_m = n^2 \sum_{\ell=1}^{m} (n-\ell)^{-1}\,
                     \mathrm{tr}\!\left(\hat\Gamma_\ell'\hat\Gamma_0^{-1}
                     \hat\Gamma_\ell\hat\Gamma_0^{-1}\right)

    which avoids forming the :math:`k^2 \times k^2` Kronecker product at
    all.

    Under the null the statistic is asymptotically :math:`\chi^2` on
    :math:`k^2(m - \mathrm{fitdf})` degrees of freedom.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Series to test, n observations x k variables. Residuals from a
        fitted model are the usual input; a transposed panel is detected
        and corrected. The columns are demeaned before use.
    lags : int, default 10
        Number of lags m in the sum. Must be smaller than n.
    cdf : callable, optional
        Null CDF of the statistic, replacing the asymptotic chi-square.
        Use it to supply a Monte Carlo null, which is more accurate at
        the small n and large m where the asymptotic approximation is
        known to be poor.
    fitdf : int, default 0
        Parameters already estimated from the series -- p + q for a
        fitted VARMA(p, q). Zero when testing a raw series rather than
        residuals. The degrees of freedom lose ``k^2 * fitdf``.
    modified : bool, default True
        Use the ``(n - lag)`` weighting of Hosking's equation (9). Set
        False for the unmodified statistic, which weights every lag by n.

    Returns
    -------
    RichResult
        keys: ``statistic``, ``p_value``, ``df``, ``lags``, ``n``, ``k``,
        ``fitdf``, ``method``.

    References
    ----------
    Hosking, J. R. M. (1980). The multivariate portmanteau statistic.
    *Journal of the American Statistical Association*, 75(371), 602-608.

    Mahdi, E. (2020). portes: an R package for portmanteau tests in time
    series models. arXiv:2005.00931, equation (9).
    """
    E = np.atleast_2d(np.asarray(X, dtype=float))
    if E.shape[0] < E.shape[1]:
        E = E.T
    n, k = E.shape
    m = int(lags)
    if m < 1:
        raise ValueError(f"lags must be at least 1, got {m}.")
    if m >= n:
        raise ValueError(f"lags must be smaller than the series length; got lags={m}, n={n}.")
    if n <= k:
        raise ValueError(f"Need more observations than variables, got n={n}, k={k}.")
    fitdf = int(fitdf)
    if fitdf < 0:
        raise ValueError(f"fitdf must not be negative, got {fitdf}.")
    if m <= fitdf:
        raise ValueError(
            f"lags must exceed fitdf, else the test has no degrees of freedom; got lags={m}, fitdf={fitdf}."
        )

    E = E - E.mean(axis=0)
    G0 = _autocov(E, 0)
    try:
        G0_inv = np.linalg.inv(G0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Lag-zero autocovariance is singular; the columns are collinear.") from exc

    total = 0.0
    for lag in range(1, m + 1):
        Gl = _autocov(E, lag)
        term = float(np.trace(Gl.T @ G0_inv @ Gl @ G0_inv))
        total += term / (n - lag) if modified else term
    statistic = n * n * total if modified else n * total

    df = k * k * (m - fitdf)
    p = float(1.0 - cdf(statistic)) if cdf is not None else float(stats.chi2.sf(statistic, df))

    return RichResult(
        title="Hosking multivariate portmanteau test",
        payload={
            "statistic": statistic,
            "p_value": p,
            "df": int(df),
            "lags": m,
            "n": int(n),
            "k": int(k),
            "fitdf": fitdf,
            "method": (
                "Hosking (1980) modified multivariate portmanteau"
                if modified
                else "Hosking (1980) multivariate portmanteau"
            ),
        },
    )


def cheatsheet():
    return "whtnse: Hosking multivariate portmanteau test for white noise"
