# morie.fn -- function file (rootcoder007/morie)
"""Moran's I test on OLS residuals with the exact moments (Cliff and Ord)."""

from ._containers import SpatialResult
from .spdurbin import residual_moran


def miols(resid, W, X=None):
    r"""Moran's I of OLS residuals with the exact moments under normality (``spdep::lm.morantest``).

    With ``M = I - X(X'X)^{-1}X'`` the residual Moran's I has ``E[I] = (n/S0)
    tr(MW) / (n - k)`` and ``Var[I] = (n/S0)^2 [tr(MWMW') + tr((MW)^2) +
    tr(MW)^2] / ((n - k)(n - k + 2)) - E[I]^2``; p-value one-sided (greater).
    ``X`` is the design matrix of the regression that produced ``resid``
    (default: intercept only). Since ``M resid = resid`` for OLS residuals the
    function delegates to :func:`morie.fn.spdurbin.residual_moran` with ``y =
    resid``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> round(miols([1.0, -2.0, 0.5, 0.5], W).expected, 12)
    -0.333333333333
    """
    e = [float(v) for v in (resid.tolist() if hasattr(resid, "tolist") else resid)]
    if X is None:
        X = [[1.0] for _ in e]
    Xm = [
        [float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in (X.tolist() if hasattr(X, "tolist") else X)
    ]
    r = residual_moran(e, Xm, W)
    return SpatialResult(
        name="miols",
        statistic=r["I"],
        p_value=r["pvalue"],
        expected=r["expected"],
        variance=r["variance"],
        extra={"z": r["z"]},
    )


miols_fn = miols


def cheatsheet() -> str:
    return "miols(resid, W, X=None) -> residual Moran's I with exact OLS moments (spdep::lm.morantest)."
