"""Wishart and inverse Wishart densities.

Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis, 3rd ed. Wiley, ch. 7.
"""

import math

from ._mvcore import as_matrix, cholesky, inverse_spd, log_mvgamma, logdet_chol, trace_prod
from ._richresult import RichResult

__all__ = ["wishartdens"]


def wishartdens(W, df, S, inverse=False):
    r"""Wishart_p(nu, S): log f(W) = ((nu - p - 1)/2) log|W| - tr(S^{-1} W)/2 - (nu p/2) log 2 - (nu/2) log|S| - log Gamma_p(nu/2).

    With ``inverse=True`` the inverse Wishart IW_p(nu, S):
    log f(W) = (nu/2) log|S| - (nu p/2) log 2 - log Gamma_p(nu/2) - ((nu + p + 1)/2) log|W| - tr(S W^{-1})/2.

    Parameters
    ----------
    W : p x p positive-definite matrix
    df : float
        nu > p - 1.
    S : p x p positive-definite scale matrix
    inverse : bool

    Returns
    -------
    RichResult
        Keys: pdf, logpdf.

    References
    ----------
    Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis, 3rd ed., ch. 7.
    Matches ``MCMCpack::dwish(W, v, S)`` and ``MCMCpack::diwish(W, v, S)``.

    Examples
    --------
    >>> round(wishartdens([[1.0]], 3, [[1.0]])["pdf"], 12)
    0.241970724519
    """
    Wm = as_matrix(W)
    Sm = as_matrix(S)
    p = len(Wm)
    if len(Sm) != p or not df > p - 1:
        raise ValueError("W and S must have the same dimension and df > p - 1")
    ldw = logdet_chol(cholesky(Wm))
    lds = logdet_chol(cholesky(Sm))
    if inverse:
        lp = (
            df / 2 * lds
            - df * p / 2 * math.log(2)
            - log_mvgamma(df / 2, p)
            - (df + p + 1) / 2 * ldw
            - trace_prod(Sm, inverse_spd(Wm)) / 2
        )
    else:
        lp = (
            (df - p - 1) / 2 * ldw
            - trace_prod(inverse_spd(Sm), Wm) / 2
            - df * p / 2 * math.log(2)
            - df / 2 * lds
            - log_mvgamma(df / 2, p)
        )
    return RichResult(
        title="Inverse Wishart" if inverse else "Wishart",
        summary_lines=[("logpdf", lp)],
        payload={"pdf": math.exp(lp), "logpdf": lp},
    )


def cheatsheet():
    return "wishartdens: Wishart and inverse Wishart log-densities."
