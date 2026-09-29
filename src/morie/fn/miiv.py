# morie.fn -- function file (rootcoder007/morie)
"""Anselin-Kelejian (1997) Moran-type test on instrumental-variables residuals."""

from ._containers import SpatialResult
from ._qpcore import inverse, ssum
from ._rrng_core import pchisq


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def miiv(resid, W, Z, H):
    r"""Anselin-Kelejian (1997) test for spatial error autocorrelation in 2SLS residuals.

    For a model estimated by two-stage least squares with regressors ``Z``
    (exogenous and endogenous, e.g. ``[X, Wy]`` in the spatial lag model) and
    instruments ``H``, with residuals ``e``, ``sigma2 = e'e / n``, ``T =
    tr((W' + W) W)`` and ``V = (Z'H (H'H)^{-1} H'Z)^{-1}``,

    ``I = (n / S0) e'We / e'e``,
    ``phi2 = [T + 4 (e'WZ) V (e'WZ)' / sigma2] / (n (S0/n)^2)`` and
    ``AK = n I^2 / phi2``, asymptotically chi-square with 1 df.

    Matches ``spreg.AKtest(case="gen")`` of PySAL.

    References
    ----------
    Anselin, L. and Kelejian, H. H. (1997). Testing for spatial error
    autocorrelation in the presence of endogenous regressors.
    *International Regional Science Review* 20, 153-182.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> Z = [[1, 0.2], [1, 1.1], [1, 1.9], [1, 3.2]]
    >>> H = [[1, 0.0, 1.0], [1, 1.0, 0.5], [1, 2.0, 2.5], [1, 3.0, 2.0]]
    >>> r = miiv([0.3, -0.5, 0.4, -0.2], W, Z, H)
    >>> round(r.statistic, 9)
    2.716591122
    """
    e = [float(v) for v in (resid.tolist() if hasattr(resid, "tolist") else resid)]
    Wm, Zm, Hm = _mat(W), _mat(Z), _mat(H)
    n, q, h = len(e), len(Zm[0]), len(Hm[0])
    s0 = ssum(v for r in Wm for v in r)
    We = [ssum(Wm[i][j] * e[j] for j in range(n)) for i in range(n)]
    ete = ssum(v * v for v in e)
    mi = n / s0 * ssum(e[i] * We[i] for i in range(n)) / ete
    sig2n = ete / n
    T = ssum(Wm[j][i] * Wm[j][i] for i in range(n) for j in range(n)) + ssum(
        Wm[i][j] * Wm[j][i] for i in range(n) for j in range(n)
    )
    HtHi = inverse([[ssum(r[a] * r[b] for r in Hm) for b in range(h)] for a in range(h)])
    ZtH = [[ssum(Zm[i][a] * Hm[i][b] for i in range(n)) for b in range(h)] for a in range(q)]
    F = [[ssum(ZtH[a][c] * HtHi[c][b] for c in range(h)) for b in range(h)] for a in range(q)]
    V = inverse([[ssum(F[a][c] * ZtH[b][c] for c in range(h)) for b in range(q)] for a in range(q)])
    eW = [ssum(e[i] * Wm[i][j] for i in range(n)) for j in range(n)]
    g = [ssum(eW[j] * Zm[j][a] for j in range(n)) for a in range(q)]
    quad = ssum(g[a] * ssum(V[a][b] * g[b] for b in range(q)) for a in range(q))
    phi2 = (T + 4.0 / sig2n * quad) / ((s0 / n) ** 2 * n)
    ak = n * mi * mi / phi2
    p = pchisq(ak, 1, lower_tail=False)
    return SpatialResult(name="miiv", statistic=ak, p_value=float(p), extra={"moran_i": mi, "phi2": phi2, "T": T})


miiv_fn = miiv


def cheatsheet() -> str:
    return "miiv(resid, W, Z, H) -> Anselin-Kelejian (1997) AK test for spatial error in 2SLS residuals."
