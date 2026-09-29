# morie.fn -- function file (rootcoder007/morie)
"""Spatial count overdispersion test."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult
from ._rrng_core import pnorm


def scdisp(y, X, trafo=None):
    r"""Spatial count overdispersion test.

    Cameron and Trivedi's (1990) regression-based test of equidispersion
    after a Poisson GLM (log link, design ``X`` with its intercept): with
    fitted means ``mu`` and ``a_i = ((y_i - mu_i)^2 - y_i) / mu_i``,

    - ``trafo=None``: ``Var y = (1 + alpha) mu``; ``alpha`` is the OLS mean
      of ``a`` and ``z = alpha / (sd(a) / sqrt(n))``;
    - ``trafo=2``: ``Var y = mu + alpha mu^2`` (NB2); ``alpha`` from the
      no-intercept OLS of ``a`` on ``mu``, ``z`` its t ratio.

    The p-value is one-sided (overdispersion, ``alpha > 0``), as
    ``AER::dispersiontest``; the reported ``dispersion`` is ``1 + alpha``
    for ``trafo=None``. Use it on the Poisson part of a spatial count model
    before choosing between :func:`morie.fn.spcount.sar_poisson` and
    ``sar_negbin``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Counts.
    X : array-like, shape (n, p)
        Design matrix including the intercept column.
    trafo : None or 2
        Alternative variance function.

    Returns
    -------
    SpatialResult
        ``statistic`` z, ``p_value``; ``extra`` has ``alpha`` and
        ``dispersion``.

    References
    ----------
    Cameron, A. C. and Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson
    model. *Journal of Econometrics*, 46(3), 347-364.

    Examples
    --------
    >>> X = [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9], [1, 0.3], [1, 0.7], [1, 0.2], [1, 0.8]]
    >>> r = scdisp([0, 3, 1, 9, 0, 6, 2, 1], X)
    >>> round(r.statistic, 10), round(r.extra["dispersion"], 10)
    (0.7182704776, 1.3317148158)
    """
    yv, Xm = gv.vec(y), gv.mat(X)
    n = len(yv)
    mu = gv.glm_fit(Xm, yv, "poisson")["fitted"]
    a = [((t - m) ** 2 - t) / m for t, m in zip(yv, mu)]
    if trafo is None:
        alpha = gv.ssum(a) / n
        se = math.sqrt(gv.ssum((v - alpha) ** 2 for v in a) / (n - 1) / n)
        disp = 1.0 + alpha
    elif trafo == 2:
        smm = gv.ssum(m * m for m in mu)
        alpha = gv.ssum(u * m for u, m in zip(a, mu)) / smm
        se = math.sqrt(gv.ssum((u - alpha * m) ** 2 for u, m in zip(a, mu)) / (n - 1) / smm)
        disp = alpha
    else:
        raise ValueError("trafo must be None or 2")
    z = alpha / se
    return SpatialResult(
        name="scdisp", statistic=z, p_value=1.0 - float(pnorm(z)), extra={"alpha": alpha, "dispersion": disp}
    )


scdisp_fn = scdisp


def cheatsheet() -> str:
    return "scdisp(y, X, trafo) -> Cameron-Trivedi overdispersion z test"
