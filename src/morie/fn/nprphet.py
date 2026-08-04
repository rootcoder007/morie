# morie.fn -- function file (rootcoder007/morie)
"""NeuralProphet decomposition, linear AR-Net."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["neural_prophet"]


def neural_prophet(ds, y, ar_layers=0, n_changepoints=3, seasonality=(365.25, 3)):
    """Additive forecast decomposition: trend, seasonality, autoregression.

    NeuralProphet keeps Prophet decomposition but replaces the sampled
    posterior with a fitted model, and adds an autoregression block.
    With the linear AR-Net the whole thing collapses to one design
    matrix, which is what is implemented here: piecewise-linear trend,
    Fourier seasonality and p lags all estimated in a single least
    squares.  A stochastically trained deep AR block would not survive
    a cross-language parity check, and the paper own linear case is not
    an approximation to it -- it is a configuration the paper defines.

    Formula: ``y_hat(t) = T(t) + S(t) + A(t)`` with
    ``T(t) = (delta_0 + Gamma(t)' delta) t + (rho_0 + Gamma(t)' rho)``,
    ``S_p(t) = sum_j a_j cos(2 pi j t / p) + b_j sin(2 pi j t / p)``,
    and linear AR ``A(t) = sum_i w_i y(t - i)``.

    Parameters
    ----------
    ds : array-like, shape (n,)
        Time index, numeric and increasing.
    y : array-like, shape (n,)
        Observed series.
    ar_layers : int, default 0
        Number of autoregressive lags p (the linear AR-Net order).
    n_changepoints : int, default 3
        Interior trend changepoints, placed at equally spaced quantiles
        of ``ds``.
    seasonality : tuple, default (365.25, 3)
        ``(period, n_fourier_terms)``; pass ``(period, 0)`` for none.

    Returns
    -------
    RichResult
        ``estimate`` (one-step-ahead fitted value at the last point),
        ``coef``, ``fitted``, ``resid``, ``rmse``, ``n``.

    References
    ----------
    Triebe, O., Hewamalage, H., Pilyugina, P., Laptev, N., Bergmeir, C.
    & Rajagopal, R. (2021).  NeuralProphet: explainable forecasting at
    scale.  arXiv:2111.15397.  Fetched; equations (1), (3), (4) and (6)
    above are quoted from that paper.
    """
    t = C.vec(ds)
    yv = C.vec(y)
    n = len(yv)
    p = int(ar_layers)
    period, nf = float(seasonality[0]), int(seasonality[1])
    cps = [S.quantile7(t, (k + 1.0) / (n_changepoints + 1.0)) for k in range(int(n_changepoints))]
    rows, targ = [], []
    for i in range(p, n):
        r = [1.0, t[i]]
        for cp in cps:
            r.append(max(t[i] - cp, 0.0))
        for j in range(1, nf + 1):
            r.append(math.cos(2.0 * math.pi * j * t[i] / period))
            r.append(math.sin(2.0 * math.pi * j * t[i] / period))
        for k in range(1, p + 1):
            r.append(yv[i - k])
        rows.append(r)
        targ.append(yv[i])
    beta, fitted, resid, _ = C.lstsq(rows, targ)
    rmse = math.sqrt(sum(v * v for v in resid) / len(resid))
    return RichResult(payload={
        "estimate": fitted[-1], "coef": beta, "fitted": fitted, "resid": resid,
        "rmse": rmse, "n": n,
        "method": "NeuralProphet decomposition with linear AR-Net"})


neuralprophet = neural_prophet


def cheatsheet():
    return "nprphet: NeuralProphet decomposition, linear AR-Net."
