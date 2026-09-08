# morie.fn -- function file (rootcoder007/morie)
"""Engle's ARCH-LM test for conditional heteroskedasticity."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["vol_engle_lagrange"]


def vol_engle_lagrange(r, q=1, demean=True):
    r"""Engle's Lagrange-multiplier test for ARCH effects.

    Regress the squared (demeaned) series on a constant and its own q
    lags,

    .. math:: e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + \dots
              + \alpha_q e_{t-q}^2 + u_t,

    and form :math:`LM = n R^2`, which is asymptotically
    :math:`\chi^2_q` under the null of no ARCH (all
    :math:`\alpha_i = 0`). Rejection says the variance is predictable
    from its own past -- the defining feature of ARCH -- while the
    *level* of the series may remain serially uncorrelated.

    This replaces a placeholder that computed a KS normality statistic
    and silently ignored ``q``. A KS test cannot detect ARCH at all: a
    GARCH process with a near-Gaussian unconditional distribution
    passes a normality check while being strongly conditionally
    heteroskedastic.

    Parameters
    ----------
    r : array-like, shape (n,)
        The series (typically returns or regression residuals).
    q : int, default 1
        Number of lags in the auxiliary regression.
    demean : bool, default True
        Subtract the sample mean before squaring.

    Returns
    -------
    RichResult
        keys: ``statistic`` (LM = m R^2 with m = n - q), ``p_value``,
        ``df``, ``r2``, ``n``, ``q``, ``method``.

    References
    ----------
    Engle, R. F. (1982). Autoregressive conditional heteroscedasticity
    with estimates of the variance of United Kingdom inflation.
    *Econometrica*, 50(4), 987-1007. Sec. 8 (the LM test).
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    q = int(q)
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}.")
    if n < q + 2:
        raise ValueError(f"Need at least q + 2 = {q + 2} observations, got {n}.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite.")

    e = r - r.mean() if demean else r
    e2 = e**2

    # Auxiliary regression of e2_t on a constant and q lags of itself.
    Y = e2[q:]
    X = np.column_stack([np.ones(n - q)] + [e2[q - j - 1 : n - j - 1] for j in range(q)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    tss = float(((Y - Y.mean()) ** 2).sum())
    if tss <= 0:
        raise ValueError("squared series has zero variance; LM test undefined.")
    r2 = 1.0 - float((resid**2).sum()) / tss

    m = n - q  # effective sample size of the auxiliary regression
    lm = m * r2
    p = float(stats.chi2.sf(lm, q))

    return RichResult(
        payload={
            "statistic": float(lm),
            "p_value": p,
            "df": q,
            "r2": float(r2),
            "n": int(n),
            "q": q,
            "method": f"Engle ARCH-LM test (q={q})",
        }
    )


def cheatsheet():
    return "volengle: Engle ARCH-LM test (n R^2 vs chi^2_q)"
