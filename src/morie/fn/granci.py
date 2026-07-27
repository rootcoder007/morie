# morie.fn -- function file (rootcoder007/morie)
"""Granger causality as conditional mutual information."""

import numpy as np
from scipy import stats

from ._richresult import RichResult
from .ggrcst import _lag_design, _rss

__all__ = ["granger_causality_info"]


def granger_causality_info(x, y, lag=1):
    r"""Gaussian conditional mutual information I(Y_t ; X_past | Y_past).

    For jointly Gaussian series the CMI has the closed form

    .. math:: I = \tfrac12 \ln
              \frac{\mathrm{RSS}_r}{\mathrm{RSS}_u}
              \;=\; \tfrac12 \,\mathcal{F},

    half the log-likelihood-ratio Granger statistic; Barnett, Barrett
    and Seth show this equals the transfer entropy exactly in the
    Gaussian case. The likelihood-ratio form :math:`2 m I \sim
    \chi^2_p` gives the p-value.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Candidate cause and response.
    lag : int, default 1
        Number of past values conditioned on.

    Returns
    -------
    RichResult
        keys: ``mi`` (nats), ``statistic`` (2 m I), ``p_value``,
        ``df``, ``n``, ``lag``, ``method``.

    References
    ----------
    Barnett, L., Barrett, A. B. & Seth, A. K. (2009). Granger
    causality and transfer entropy are equivalent for Gaussian
    variables. *Physical Review Letters*, 103(23), 238701.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have equal length.")
    p = int(lag)
    if p < 1:
        raise ValueError(f"lag must be at least 1, got {p}.")
    n = y.size
    m = n - p
    if m - 2 * p - 1 < 1:
        raise ValueError(f"need at least {3 * p + 2} observations for lag = {p}, got {n}.")

    Dr, t = _lag_design(y, x, p, include_x=False)
    Du, _ = _lag_design(y, x, p, include_x=True)
    rss_r, rss_u = _rss(Dr, t), _rss(Du, t)
    if rss_u <= 0:
        raise ValueError("unrestricted model fits exactly; CMI undefined.")
    mi = 0.5 * np.log(rss_r / rss_u)
    lr = 2.0 * m * mi
    pv = float(stats.chi2.sf(lr, p))

    return RichResult(
        payload={
            "mi": float(mi),
            "statistic": float(lr),
            "p_value": pv,
            "df": p,
            "n": int(n),
            "lag": p,
            "method": "Granger causality as Gaussian CMI (0.5 ln RSS_r/RSS_u)",
        }
    )


def cheatsheet():
    return "granci: I(Y_t; X_past | Y_past) = 0.5 ln(RSS_r/RSS_u); 2mI ~ chi2_p"
