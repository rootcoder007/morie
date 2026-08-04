"""Partial autocorrelation via the Levinson-Durbin recursion."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["partial_autocorrelation"]


def partial_autocorrelation(y, lag_max):
    r"""Sample partial autocorrelation function.

    The PACF at lag k is the last coefficient :math:`\phi_{kk}` of the
    order-k autoregression, obtained from the sample autocorrelations by
    the Levinson-Durbin recursion:

    .. math::
        \phi_{kk} = \frac{r_k - \sum_{j=1}^{k-1}\phi_{k-1,j} r_{k-j}}
                         {1 - \sum_{j=1}^{k-1}\phi_{k-1,j} r_j}

    with :math:`\phi_{kj} = \phi_{k-1,j} - \phi_{kk}\phi_{k-1,k-j}` and
    :math:`\phi_{11} = r_1`.

    The autocorrelations use the divide-by-n (biased) convention, which
    is what R's ``acf`` and ``pacf`` use, so the sequence is guaranteed
    positive semi-definite and the recursion cannot divide by zero for a
    non-degenerate series.

    This body previously read ``stats.spearmanr(y[:n], y[:n])`` -- the
    series correlated with itself, so ``statistic`` was identically 1.0
    for every input and every lag. It is one of the generator's pasted
    templates and had nothing to do with the documented method. The same
    template was found in ``acsamp``, ``joacf`` and ``jopacf``.

    Parameters
    ----------
    y : array-like
        The series.
    lag_max : int
        Highest lag to report; must be at least 1 and less than ``len(y)``.

    Returns
    -------
    RichResult
        ``pacf`` (lags 1..lag_max), ``acf`` (lags 0..lag_max), ``n``,
        and ``phi`` -- the final order-``lag_max`` AR coefficients.

    References
    ----------
    Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015).
    *Time Series Analysis: Forecasting and Control*, 5th ed., §3.2.6.
    Durbin, J. (1960). The fitting of time-series models.
    *Revue de l'Institut International de Statistique*, 28(3), 233-244.
    """
    vals = [float(v) for v in np.asarray(y, dtype=float).ravel().tolist()]
    n = len(vals)
    k_max = int(lag_max)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if k_max < 1 or k_max >= n:
        raise ValueError("lag_max must satisfy 1 <= lag_max < len(y); "
                         "got %r with n=%d" % (lag_max, n))

    mean = sum(vals) / n
    dev = [v - mean for v in vals]
    c0 = sum(d * d for d in dev) / n
    if c0 <= 0.0:
        raise ValueError("series is constant; the PACF is undefined.")

    # divide by n, not n - k: R's convention, and it keeps the
    # autocovariance sequence positive semi-definite
    r = [1.0]
    for k in range(1, k_max + 1):
        ck = sum(dev[t] * dev[t + k] for t in range(n - k)) / n
        r.append(ck / c0)

    pacf = []
    phi_prev = []
    for k in range(1, k_max + 1):
        if k == 1:
            phi_kk = r[1]
            phi_cur = [phi_kk]
        else:
            num = r[k] - sum(phi_prev[j - 1] * r[k - j] for j in range(1, k))
            den = 1.0 - sum(phi_prev[j - 1] * r[j] for j in range(1, k))
            if abs(den) < 1e-300:
                raise ValueError("Levinson-Durbin denominator vanished at "
                                 "lag %d; the series is degenerate." % k)
            phi_kk = num / den
            phi_cur = [phi_prev[j - 1] - phi_kk * phi_prev[k - j - 1]
                       for j in range(1, k)]
            phi_cur.append(phi_kk)
        pacf.append(float(phi_kk))
        phi_prev = phi_cur

    return RichResult(
        payload={
            "pacf": pacf,
            "acf": [float(v) for v in r],
            "phi": [float(v) for v in phi_prev],
            "lag_max": k_max,
            "n": n,
            "method": "Partial autocorrelation, Levinson-Durbin recursion",
        }
    )


def cheatsheet():
    return {
        "name": "partial_autocorrelation",
        "what": "PACF via Levinson-Durbin; phi_kk is the order-k AR "
                "coefficient at lag k",
        "returns": "pacf, acf, phi, lag_max, n",
        "note": "divide-by-n autocorrelations, matching R's acf/pacf",
    }


pacfp = partial_autocorrelation
partialautocorrelation = partial_autocorrelation
