# morie.fn -- function file (rootcoder007/morie)
"""MGWR sigma-squared estimate."""

from ._containers import SpatialResult
from ._qpcore import ssum


def mgwrsig(resid, tr_S, n=None):
    r"""Error variance of an (M)GWR fit, ``sigma^2 = RSS / (n - tr S)``.

    The estimator used for MGWR standard errors (Yu et al. 2020;
    ``GWmodel::gwr.multiscale``'s ``sigma.hat``); ``n`` defaults to the
    number of residuals.

    References
    ----------
    Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. and Wolf, L. J.
    (2020). Inference in multiscale geographically weighted regression.
    *Geographical Analysis* 52, 87-106.

    Examples
    --------
    >>> round(mgwrsig([0.5, -0.25, 0.1, -0.3, 0.2, -0.05], 2.5).statistic, 12)
    0.13
    """
    e = [float(v) for v in resid]
    n = len(e) if n is None else int(n)
    val = ssum(v * v for v in e) / (n - float(tr_S))
    return SpatialResult(name="mgwrsig", statistic=val, extra={"rss": ssum(v * v for v in e), "trS": float(tr_S)})


mgwrsig_fn = mgwrsig


def cheatsheet() -> str:
    return "mgwrsig(resid, tr_S) -> sigma^2 = RSS / (n - tr S)."
