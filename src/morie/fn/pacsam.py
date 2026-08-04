"""Sample partial autocorrelation function via the Durbin-Levinson recursion."""

from . import _array_core as np
from ._richresult import RichResult
from .acsamp import _acvf

__all__ = ["sample_partial_autocorr"]


def _durbin_levinson(r, max_lag):
    """PACF phi_kk for k = 1..max_lag from autocorrelations r[0..max_lag]."""
    phi = [0.0] * (max_lag + 1)      # phi[k] holds phi_kk
    prev = []                        # coefficients of the order-(k-1) fit
    for k in range(1, max_lag + 1):
        num = r[k] - sum(prev[j] * r[k - 1 - j] for j in range(k - 1))
        den = 1.0 - sum(prev[j] * r[j + 1] for j in range(k - 1))
        if den == 0:
            raise ValueError("singular Durbin-Levinson step; series is degenerate.")
        kk = num / den
        cur = [prev[j] - kk * prev[k - 2 - j] for j in range(k - 1)] + [kk]
        phi[k] = kk
        prev = cur
    return phi[1:]


def sample_partial_autocorr(y, max_lag=20):
    r"""Sample partial autocorrelation function.

    :math:`\phi_{kk}` is the correlation between :math:`y_t` and
    :math:`y_{t-k}` once the intervening lags are projected out. It is
    obtained from the sample autocorrelations by the Durbin-Levinson
    recursion

    .. math::

       \phi_{kk} = \frac{r_k - \sum_{j=1}^{k-1}\phi_{k-1,j}\,r_{k-j}}
                        {1 - \sum_{j=1}^{k-1}\phi_{k-1,j}\,r_j},
       \qquad
       \phi_{kj} = \phi_{k-1,j} - \phi_{kk}\,\phi_{k-1,k-j},

    which solves the Yule-Walker system in :math:`O(k^2)` rather than
    inverting a Toeplitz matrix at every order.

    Consistency check built into the definition: :math:`\phi_{11}=r_1`
    exactly, and for an AR(p) process the PACF cuts off after lag p --
    which is the whole reason it is looked at alongside the ACF.

    Parameters
    ----------
    y : array-like
        The series.
    max_lag : int
        Highest lag to return; clipped to n-1.

    Returns
    -------
    RichResult
        Keys ``pacf`` (list, index 0 is lag 1), ``lags``, ``n``,
        ``max_lag``, ``ci_bound``.

    References
    ----------
    Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis:
    Forecasting and Control*, rev. ed., Holden-Day, sec. 3.2.6.
    Durbin, J. (1960). The fitting of time series models.
    *Revue de l'Institut International de Statistique*, 28, 233-244.
    """
    v = [float(t) for t in np.asarray(y, dtype=float).ravel().tolist()]
    n = len(v)
    if n < 3:
        raise ValueError("need at least 3 observations.")
    max_lag = min(int(max_lag), n - 1)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    c = _acvf(v, n, max_lag)
    if c[0] <= 0:
        raise ValueError("series is constant; the autocorrelation is undefined.")
    r = [ck / c[0] for ck in c]
    p = _durbin_levinson(r, max_lag)
    return RichResult(
        payload={
            "pacf": p,
            "lags": list(range(1, max_lag + 1)),
            "acf": r,
            "n": n,
            "max_lag": max_lag,
            "ci_bound": 1.96 / (n ** 0.5),
            "method": "Sample PACF via the Durbin-Levinson recursion",
        }
    )


def cheatsheet():
    return "pacsam: sample partial autocorrelation phi_kk, k = 1 .. max_lag"
