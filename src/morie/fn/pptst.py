# morie.fn -- function file (rootcoder007/morie)
"""Phillips-Perron unit root test."""

from __future__ import annotations

from . import _array_core as np

from ._containers import TestResult

__all__ = ["pptst", "pp_test"]


def pp_test(y, lags: int | None = None) -> TestResult:
    """Phillips-Perron unit root test.

    Nonparametric correction for heteroskedasticity and autocorrelation in the
    residuals of the Dickey-Fuller regression.  Robust to unknown forms of
    serial correlation.

    Tests H0: unit root (series is I(1)).

    Parameters
    ----------
    y : array-like
        Univariate time series (n,).
    lags : int or None
        Lag truncation for the Newey-West long-run variance. If None uses
        ``trunc(4 * (n/100)^(1/4))`` with n = len(y) - 1, urca's "short".

    Returns
    -------
    TestResult
        statistic: PP Z_tau test statistic.
        p_value: approximate p-value binned from MacKinnon (1994) critical
            values for regression with constant.
        extra['critical_values']: dict with 1%, 5%, 10% levels.
        extra['lags']: lag truncation used.
        extra['lrvar']: estimated long-run variance.

    Notes
    -----
    Follows urca::ur.pp(type = "Z-tau", model = "constant"): regress
    y_t on (1, y_{t-1}) over the n = T - 1 pairs; with s = sum(u^2)/n,
    the Bartlett long-run variance
    sig = s + (2/n) sum_l (1 - l/(L+1)) sum_t u_t u_{t-l},
    and ybar2 = sum((y_t - mean y)^2) / n^2,

        Z_tau = sqrt(s/sig) * t_rho - (sig - s) / (2 sig) * sqrt(sig / ybar2),

    t_rho = (rho - 1) / se(rho). Critical values are MacKinnon's
    finite-sample response surfaces as urca uses them.

    Examples
    --------
    A stationary sawtooth, x_t = (7t mod 11) - 5: urca's
    ur.pp(x, type = "Z-tau", model = "constant", lags = "short") gives
    Z_tau = -12.65008 with 3 lags, far beyond the 1% value:

    >>> r = pp_test([float((7 * t) % 11) - 5.0 for t in range(60)])
    >>> round(r.statistic, 5), r.extra["lags"], r.p_value
    (-12.65008, 3, 0.01)

    References
    ----------
    Phillips P.C.B. & Perron P. (1988). Testing for a unit root in time series
    regression. Biometrika, 75(2), 335-346.

    MacKinnon J.G. (1994). Approximate asymptotic distribution functions for
    unit root and cointegration tests.
    Journal of Business & Economic Statistics, 12(2), 167-176.

    Pfaff, B. (2008). urca: Unit root and cointegration tests for time
    series data, function ur.pp.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if T < 4:
        raise ValueError(f"Need >= 4 observations, got {T}.")
    n = T - 1
    if lags is None:
        lags = int(4.0 * (n / 100.0) ** 0.25)
    lags = max(0, min(int(lags), n - 2))
    ycur = [float(v) for v in y[1:]]
    ylag = [float(v) for v in y[:-1]]
    ml = sum(ylag) / n
    mc = sum(ycur) / n
    sxx = sum((v - ml) ** 2 for v in ylag)
    nan_out = TestResult(test_name="Phillips-Perron", statistic=np.nan,
                         p_value=np.nan, n=T, method=f"PP Z_tau (lags={lags})",
                         extra={"critical_values": {}, "lags": lags, "lrvar": np.nan})
    if n - 2 <= 0 or sxx <= 0.0:
        return nan_out
    rho = sum((a - ml) * (c - mc) for a, c in zip(ylag, ycur)) / sxx
    mu = mc - rho * ml
    res = [c - mu - rho * a for a, c in zip(ylag, ycur)]
    ssr = sum(u * u for u in res)
    se_rho = (ssr / (n - 2) / sxx) ** 0.5
    if se_rho <= 0.0:
        return nan_out
    t_rho = (rho - 1.0) / se_rho
    s = ssr / n
    lrvar = s
    for l in range(1, lags + 1):
        lrvar += (2.0 / n) * (1.0 - l / (lags + 1.0)) * sum(
            res[t] * res[t - l] for t in range(l, n))
    ybar2 = sum((c - mc) ** 2 for c in ycur) / n ** 2
    if lrvar <= 0.0 or ybar2 <= 0.0:
        return nan_out
    lam = 0.5 * (lrvar - s) / lrvar
    z_tau = (s / lrvar) ** 0.5 * t_rho - lam * lrvar ** 0.5 / ybar2 ** 0.5
    cv = {"1%": -3.4335 - 5.999 / n - 29.25 / n ** 2,
          "5%": -2.8621 - 2.738 / n - 8.36 / n ** 2,
          "10%": -2.5671 - 1.438 / n - 4.48 / n ** 2}
    if z_tau <= cv["1%"]:
        pval = 0.01
    elif z_tau <= cv["5%"]:
        pval = 0.05
    elif z_tau <= cv["10%"]:
        pval = 0.10
    else:
        pval = 0.90

    return TestResult(
        test_name="Phillips-Perron",
        statistic=float(z_tau),
        p_value=float(pval),
        n=T,
        method=f"PP Z_tau (lags={lags})",
        extra={"critical_values": cv, "lags": lags, "lrvar": lrvar},
    )


pptst = pp_test


def cheatsheet() -> str:
    return "pp_test(y, lags=None) -> Phillips-Perron unit root test."
