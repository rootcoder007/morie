# morie.fn -- function file (rootcoder007/morie)
"""Spatial heterogeneity test (Breusch-Pagan spatial)."""

from ._containers import TestResult
from ._qpcore import inverse, ssum
from ._rrng_core import pchisq
from .lmdiag import _flat, _mat, _mv


def spatial_heterogeneity(residuals, W, Z=None):
    r"""Studentised Breusch-Pagan (Koenker 1981) test of heteroskedasticity along a spatial pattern.

    The squared residuals ``e^2`` are regressed on ``[1, Z]`` and the statistic
    is ``n R^2``, asymptotically chi-square with ``ncol(Z)`` df (Breusch and
    Pagan 1979; Koenker 1981; the spatial form of Anselin 1988, sec. 8.2).
    ``Z`` defaults to the spatial lag of the squared residuals ``W e^2``,
    i.e. the test asks whether the error variance at a site co-moves with the
    variance of its neighbours; supply ``Z`` (e.g. coordinates or regional
    dummies) for other spatial variance patterns. Equals
    ``lmtest::bptest(e^2 ~ Z, studentize = TRUE)`` applied to the residuals.

    References
    ----------
    Breusch, T. S. and Pagan, A. R. (1979). A simple test for
    heteroscedasticity and random coefficient variation. *Econometrica* 47,
    1287-1294.
    Koenker, R. (1981). A note on studentizing a test for heteroscedasticity.
    *Journal of Econometrics* 17, 107-112.
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = spatial_heterogeneity([0.5, -1.0, 2.0, -0.2, 0.3], W)
    >>> round(r.statistic, 10)
    0.3438837925
    """
    e = _flat(residuals)
    n = len(e)
    Wm = _mat(W)
    if len(Wm) != n or len(Wm[0]) != n:
        raise ValueError("W must be (n, n)")
    e2 = [v * v for v in e]
    Zm = [[v] for v in _mv(Wm, e2)] if Z is None else _mat(Z)
    D = [[1.0] + r for r in Zm]
    k = len(D[0])
    b = _mv(
        inverse([[ssum(r[a] * r[c] for r in D) for c in range(k)] for a in range(k)]),
        [ssum(D[i][a] * e2[i] for i in range(n)) for a in range(k)],
    )
    fit = _mv(D, b)
    m = ssum(e2) / n
    ssr = ssum((e2[i] - fit[i]) ** 2 for i in range(n))
    sst = ssum((v - m) ** 2 for v in e2)
    stat = n * (1.0 - ssr / sst)
    df = k - 1
    return TestResult(
        test_name="spatial_heterogeneity",
        statistic=stat,
        p_value=float(pchisq(stat, df, lower_tail=False)),
        df=float(df),
        method="studentised Breusch-Pagan (Koenker)",
        n=n,
    )


sphet = spatial_heterogeneity


def cheatsheet() -> str:
    return "spatial_heterogeneity(residuals, W, Z=None) -> Koenker n R^2 of e^2 on [1, Z] (default Z = W e^2)."
