# morie.fn -- function file (rootcoder007/morie)
"""Phillips-Perron unit root test (HAC-corrected)."""

from __future__ import annotations

import math

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["phillips_perron_unit_root"]

# Dickey-Fuller tables for the trend-included regression, as carried by
# tseries::pp.test.  Rows are sample sizes 25, 50, 100, 250, 500, inf;
# columns are the probabilities in _PP_P.  Values are negated on use.
_PP_T = (25.0, 50.0, 100.0, 250.0, 500.0, 100000.0)
_PP_P = (0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99)
_PP_ALPHA = (
    (22.5, 19.9, 17.9, 15.6, 3.66, 2.51, 1.53, 0.43),
    (25.7, 22.4, 19.8, 16.8, 3.71, 2.60, 1.66, 0.65),
    (27.4, 23.6, 20.7, 17.5, 3.74, 2.62, 1.73, 0.75),
    (28.4, 24.4, 21.3, 18.0, 3.75, 2.64, 1.78, 0.82),
    (28.9, 24.8, 21.5, 18.1, 3.76, 2.65, 1.78, 0.84),
    (29.5, 25.1, 21.8, 18.3, 3.77, 2.66, 1.79, 0.87),
)
_PP_TALPHA = (
    (4.38, 3.95, 3.60, 3.24, 1.14, 0.80, 0.50, 0.15),
    (4.15, 3.80, 3.50, 3.18, 1.19, 0.87, 0.58, 0.24),
    (4.04, 3.73, 3.45, 3.15, 1.22, 0.90, 0.62, 0.28),
    (3.99, 3.69, 3.43, 3.13, 1.23, 0.92, 0.64, 0.31),
    (3.98, 3.68, 3.42, 3.13, 1.24, 0.93, 0.65, 0.32),
    (3.96, 3.66, 3.41, 3.12, 1.25, 0.94, 0.66, 0.33),
)


def _interp(xs, ys, x):
    """Linear interpolation with flat extrapolation (R ``approx`` rule 2)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def ppcritical(n, kind="Z(t_alpha)"):
    """Interpolated Dickey-Fuller critical values at sample size ``n``."""
    tab = _PP_ALPHA if kind == "Z(alpha)" else _PP_TALPHA
    cols = len(_PP_P)
    return [-_interp(_PP_T, [tab[r][c] for r in range(len(_PP_T))], float(n)) for c in range(cols)]


def phillips_perron_unit_root(x, lags=None, kind="Z(t_alpha)"):
    """Phillips-Perron test for a unit root, trend included.

    The auxiliary regression is ``y_t = mu + beta (t - n/2) + rho y_{t-1}
    + u_t`` on the ``n`` usable pairs.  With ``s^2 = sum u^2 / n`` and
    the Bartlett long-run variance

        ``lambda^2 = s^2 + (2/n) sum_{i=1}^{l} (1 - i/(l+1)) sum_t u_t u_{t-i}``

    the two statistics are

        ``Z(alpha)   = n(rho - 1) - n^6 (lambda^2 - s^2) / (24 D)``
        ``Z(t_alpha) = sqrt(s^2/lambda^2) t_rho
                       - n^3 (lambda^2 - s^2) / (4 sqrt(3) sqrt(D) lambda)``

    where ``D`` is the trend-corrected sum of squares

        ``D = n^2(n^2-1) sum y_{t-1}^2 / 12 - n (sum t y_{t-1})^2
              + n(n+1) (sum t y_{t-1})(sum y_{t-1})
              - n(n+1)(2n+1) (sum y_{t-1})^2 / 6``.

    The correction is non-parametric: the OLS ``t`` statistic is
    rescaled by the ratio of short-run to long-run variance instead of
    augmenting the regression with lagged differences, which is what
    distinguishes this from the augmented Dickey-Fuller test.  The
    default truncation lag is ``floor(4 (n/100)^{1/4})``.

    Parameters
    ----------
    x : array-like
        Series in time order.
    lags : int, optional
        Bartlett truncation lag; the short-lag rule if omitted.
    kind : {"Z(t_alpha)", "Z(alpha)"}
        Which statistic to report.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``rho``, ``lags``, ``s2``,
        ``lambda2``, ``n``, ``method``.

    References
    ----------
    Phillips and Perron (1988), Testing for a unit root in time series
    regression, Biometrika 75:335-346.  Paywalled; the coded form,
    including the ``D`` expression, the ``pp_sum`` Bartlett kernel
    (src/ppsum.c) and the Dickey-Fuller tables reproduced above, was
    read from Trapletti and Hornik's ``tseries`` package, R/test.R
    (tarball tseries_0.10-62 fetched from CRAN).  The p-value is
    interpolated in the same tables, flat beyond their range, so it is
    only meaningful inside 1%-99%.
    """
    x = T.vec(x)
    nn = len(x)
    if nn < 6:
        raise ValueError("need at least 6 observations")
    yt = x[1:]
    yt1 = x[:-1]
    n = len(yt)
    if kind not in ("Z(alpha)", "Z(t_alpha)"):
        raise ValueError("kind must be 'Z(alpha)' or 'Z(t_alpha)'")
    lag = int(lags) if lags is not None else int(4.0 * (n / 100.0) ** 0.25)
    if lag < 1:
        lag = 1
    X = [[1.0, (i + 1) - n / 2.0, yt1[i]] for i in range(n)]
    beta, fitted, u, xtxinv = T.olsfit(X, yt)
    ssqru = sum(ui * ui for ui in u) / n
    ssqrtl = T.lrvnw(u, lag)
    n2 = float(n) ** 2
    s = [i + 1.0 for i in range(n)]
    sy = sum(yt1)
    sty = sum(s[i] * yt1[i] for i in range(n))
    trm1 = n2 * (n2 - 1.0) * sum(v * v for v in yt1) / 12.0
    trm2 = n * sty * sty
    trm3 = n * (n + 1.0) * sty * sy
    trm4 = n * (n + 1.0) * (2.0 * n + 1.0) * sy * sy / 6.0
    d = trm1 - trm2 + trm3 - trm4
    rho = beta[2]
    if kind == "Z(alpha)":
        stat = n * (rho - 1.0) - (float(n) ** 6) / (24.0 * d) * (ssqrtl - ssqru)
    else:
        sigma2 = sum(ui * ui for ui in u) / (n - 3.0)
        se_rho = math.sqrt(sigma2 * xtxinv[2][2])
        tstat = (rho - 1.0) / se_rho
        stat = (math.sqrt(ssqru) / math.sqrt(ssqrtl) * tstat
                - (float(n) ** 3) / (4.0 * math.sqrt(3.0) * math.sqrt(d) * math.sqrt(ssqrtl))
                * (ssqrtl - ssqru))
    crit = ppcritical(n, kind)
    p = _interp(crit, list(_PP_P), stat)
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(p),
            "rho": float(rho),
            "lags": int(lag),
            "s2": float(ssqru),
            "lambda2": float(ssqrtl),
            "n": int(n),
            "method": f"Phillips-Perron unit root test, {kind}",
        }
    )


def cheatsheet():
    return "phillips_perron_unit_root(x, lags, kind): PP unit root, HAC-corrected DF."


# compact alias per ledger/NAMING.md
ppunitroot = phillips_perron_unit_root
