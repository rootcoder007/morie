# morie.fn -- function file (rootcoder007/morie)
"""MGWR backfitting algorithm iteration."""

from ._richresult import RichResult
from .mgwrfit import _backfit, _prep


def mgwrbk(y, X, coords, bandwidths, max_iter=10, kernel="bisquare", adaptive=False, threshold=1e-10):
    r"""Run (at most max_iter passes of) the MGWR backfitting algorithm with fixed bandwidths.

    Starting from OLS, each pass refits covariate k's smooth as the
    one-covariate GWR of the partial residual y - sum_{m != k} f_m with
    bandwidth bandwidths[k] (Buja, Hastie and Tibshirani 1989;
    Fotheringham, Yang and Kang 2017, Table 1); returns the coefficients,
    residuals, RSS, the passes run and the last change criterion is below
    threshold when converged. Predictors are not centred.

    References
    ----------
    Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
    additive models. *Annals of Statistics* 17, 453-510.
    Fotheringham, A. S., Yang, W. and Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). *Annals of the American
    Association of Geographers* 107, 1247-1265.
    Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. and Wolf, L. J.
    (2020). Inference in multiscale geographically weighted regression.
    *Geographical Analysis* 52, 87-106.

    Examples
    --------
    >>> import math
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[math.sin(i), (0.3 * i) % 1.1] for i in range(20)]
    >>> y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(20)]
    >>> r = mgwrbk(y, X, P, [6.0, 3.0, 8.0], max_iter=3, kernel="gaussian")
    >>> r["iterations"], round(r["rss"], 10)
    (3, 0.4529135225)
    """
    yv, Xm, D = _prep(y, X, coords)
    n, p = len(yv), len(Xm[0])
    beta, resid, rss, it, _, _, _ = _backfit(yv, Xm, D, bandwidths, kernel, adaptive, threshold, max_iter, False, False)
    return RichResult(
        payload={
            "betas": [[beta[k][i] for k in range(p)] for i in range(n)],
            "residuals": resid,
            "rss": rss,
            "iterations": it,
            "converged": it < max_iter,
        }
    )


mgwrbk_fn = mgwrbk


def cheatsheet() -> str:
    return "mgwrbk(y, X, coords, bandwidths, max_iter=10) -> MGWR backfitting passes with fixed bandwidths."
