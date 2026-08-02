# morie.fn -- function file (rootcoder007/morie)
"""Sample autocorrelation function."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["autocorrelation"]


def autocorrelation(y, lag_max=None, ci=0.95):
    r"""Sample ACF with significance bands and a Ljung-Box test.

    .. math::
        r_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar y)(y_{t-k} - \bar y)}
                   {\sum_{t=1}^{n}(y_t - \bar y)^2}.

    The denominator uses **all** :math:`n` terms while the numerator has only
    :math:`n-k`, which biases :math:`r_k` toward zero at long lags. That is
    deliberate: the resulting estimator is the one guaranteed to be a valid
    (positive semi-definite) autocorrelation sequence, and the unbiased
    alternative is not. Any model fitted to an invalid sequence can fail in
    ways that are hard to diagnose.

    The :math:`\pm 1.96/\sqrt n` bands assume white noise, so they answer
    "is this series uncorrelated", not "is this residual correlation
    acceptable after fitting a model" -- the latter needs the degrees of
    freedom the model consumed, which is what the Ljung-Box ``fitdf``
    argument is for.

    Reading a single significant spike among 40 lags as meaningful is a
    multiple-comparison error: at 5% about two spikes are expected by chance,
    which is why the portmanteau test is reported alongside.

    Parameters
    ----------
    y : array-like
        Series.
    lag_max : int, optional
        Highest lag. Defaults to ``min(10 log10(n), n - 1)``.
    ci : float
        Confidence level for the white-noise bands.

    Returns
    -------
    RichResult
        ``acf``, ``lags``, ``ci_bound``, ``significant``,
        ``ljung_box``, ``ljung_box_p``.

    References
    ----------
    Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time
        series models. *Biometrika*, 65(2), 297-303.
    Examples
    --------
    White noise shows no systematic autocorrelation and the portmanteau test
    does not reject.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = autocorrelation(rng.normal(size=500), lag_max=10)
    >>> bool(r["ljung_box_p"] > 0.05)
    True
    >>> float(r["acf"][0])
    1.0

    An AR(1) with rho = 0.8 decays geometrically, so lag 1 is near 0.8 and
    lag 2 near 0.64.

    >>> x = np.zeros(2000)
    >>> for i in range(1, 2000):
    ...     x[i] = 0.8 * x[i - 1] + rng.normal()
    >>> a = autocorrelation(x, lag_max=5)["acf"]
    >>> bool(abs(a[1] - 0.8) < 0.06 and abs(a[2] - 0.64) < 0.08)
    True

    Strong autocorrelation is caught by the test.

    >>> bool(autocorrelation(x, lag_max=10)["ljung_box_p"] < 0.001)
    True

    >>> autocorrelation([1.0, 2.0, 3.0], lag_max=10)
    Traceback (most recent call last):
        ...
    ValueError: lag_max must be between 1 and 2
    """
    from ._stats_core import chi2, norm

    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    if lag_max is None:
        lag_max = int(min(10 * np.log10(n), n - 1))
    lag_max = int(lag_max)
    if not 1 <= lag_max <= n - 1:
        raise ValueError(f"lag_max must be between 1 and {n - 1}")

    yc = y - y.mean()
    denom = float(np.sum(yc**2))
    # Same denominator at every lag: biased toward zero, but guarantees a
    # positive semi-definite sequence, which the unbiased version does not.
    acf = np.array([1.0] + [float(np.sum(yc[k:] * yc[:-k]) / denom)
                            for k in range(1, lag_max + 1)])
    bound = float(norm.ppf(0.5 + ci / 2.0) / np.sqrt(n))
    lb = float(n * (n + 2) * np.sum(acf[1:] ** 2 / (n - np.arange(1, lag_max + 1))))
    return RichResult(
        title="Autocorrelation function",
        summary_lines=[("n", int(n)), ("max lag", lag_max),
                       ("Ljung-Box p", float(chi2.sf(lb, lag_max)))],
        warnings=["the bands assume white noise; after fitting a model, "
                  "subtract the degrees of freedom it consumed before reading "
                  "them"],
        payload={
            "acf": acf, "lags": np.arange(lag_max + 1), "ci_bound": bound,
            "significant": np.abs(acf[1:]) > bound,
            "n_significant": int(np.sum(np.abs(acf[1:]) > bound)),
            "ljung_box": lb, "ljung_box_p": float(chi2.sf(lb, lag_max)),
            "n": int(n), "method": "autocorrelation",
        },
    )


def cheatsheet():
    return "acfP: same denominator at every lag keeps the sequence PSD; one spike in 40 is chance, read Ljung-Box"
