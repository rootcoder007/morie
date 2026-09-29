# morie.fn -- function file (rootcoder007/morie)
"""Kelejian-Prucha LM test for SAC model."""

from . import _robust_core as _rc
from .lmdiag import _design, _flat, _mat, _mv
from .miiv import miiv


def lmkp(y, X, W):
    r"""Kelejian-Prucha (2001) Moran-type test for spatial error in the spatial lag (SAC) model.

    The lag model y = rho W y + X beta + u is estimated by spatial 2SLS
    with instruments H = [X, WX, W^2 X] (Kelejian and Prucha 1998); under
    H0: lambda = 0 in the SAC model u = lambda W u + e the normalised
    squared Moran statistic of the 2SLS residuals is asymptotically chi-square
    with 1 df. Kelejian and Prucha (2001) derive its variance for models with
    endogenous regressors; with Z = [Wy, X] it takes the Anselin-Kelejian
    (1997) form computed by :func:`morie.fn.miiv.miiv`.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (2001). On the asymptotic distribution
    of the Moran I test statistic with applications. *Journal of
    Econometrics* 104, 219-257.
    Anselin, L. and Kelejian, H. H. (1997). Testing for spatial error
    autocorrelation in the presence of endogenous regressors. *International
    Regional Science Review* 20, 153-182.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0, 0, 0.5], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0],
    ...      [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0.5, 0, 0, 0, 0.5, 0]]
    >>> X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 0.5]]
    >>> r = lmkp([1.0, 2.5, 2.0, 4.5, 4.0, 6.5], X, W)
    >>> round(r.statistic, 10)
    0.4699041719
    """
    yv = _flat(y)
    n = len(yv)
    Xm, const = _design(X, n)
    Wm = _mat(W)
    xs = [[r[c] for c in range(len(r)) if c not in const] for r in Xm]
    fit = _rc.spatial_2sls(yv, xs, Wm, add_intercept=True)
    cols = [[1.0] * n] + [[r[c] for r in xs] for c in range(len(xs[0]))]
    H = list(cols)
    for c in cols[1:]:
        wc = _mv(Wm, c)
        H += [wc, _mv(Wm, wc)]
    wy = _mv(Wm, yv)
    Z = [[wy[i]] + [c[i] for c in cols] for i in range(n)]
    Hm = [[c[i] for c in H] for i in range(n)]
    r = miiv(fit["residuals"], Wm, Z, Hm)
    r.name = "lmkp"
    r.extra["rho"] = fit["rho"]
    return r


lmkp_fn = lmkp


def cheatsheet() -> str:
    return "lmkp(y, X, W) -> Kelejian-Prucha (2001) test for spatial error after spatial 2SLS of the lag model."
