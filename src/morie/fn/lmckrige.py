# morie.fn -- function file (rootcoder007/morie)
"""Multivariate geostatistics: the linear model of coregionalization (LMC; one structure is the intrinsic
coregionalization model) and ordinary or simple cokriging of heterotopic data in any dimension."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .stkrig import _comp

__all__ = ["lmc_covariance", "lmc_cokriging"]


def _rho(d, s):
    c = {k: v for k, v in s.items() if k != "B"}
    c["psill"] = 1.0
    c.pop("nugget", None)
    return _comp(d, c)


def _check(lmc):
    lmc = [lmc] if isinstance(lmc, dict) else list(lmc)
    p = len(lmc[0]["B"])
    for s in lmc:
        B = s["B"]
        if len(B) != p or any(len(r) != p for r in B):
            raise ValueError("every structure needs a p x p coregionalization matrix B")
        if any(abs(B[a][b] - B[b][a]) > 1e-12 for a in range(p) for b in range(p)):
            raise ValueError("B must be symmetric")
    return lmc, p


def lmc_covariance(h: float, lmc) -> list:
    r"""Cross-covariance matrix ``C(h) = sum_s B_s rho_s(h)`` of a linear model of coregionalization.

    ``lmc`` is a list of structures ``{"model", "range", ("kappa"), "B"}`` with
    correlation functions ``rho_s`` of :func:`morie.fn.krgsys.kriging_covariance`
    (``Exp``, ``Gau``, ``Sph``, ``Mat``; ``Nug`` acts only at ``h = 0``) and
    positive semi-definite coregionalization matrices ``B_s``. A single
    structure is the intrinsic coregionalization model ``C(h) = B rho(h)``.

    References
    ----------
    Wackernagel, H. (2003). *Multivariate Geostatistics*, 3rd edn. Springer, ch. 23-24.
    Goulard, M. and Voltz, M. (1992). Linear coregionalization model: tools
    for estimation and choice of cross-variogram matrix. *Mathematical Geology*, 24, 269-286.

    Examples
    --------
    >>> lmc = [{"model": "Exp", "range": 2.0, "B": [[1.0, 0.5], [0.5, 2.0]]}]
    >>> [[round(v, 6) for v in r] for r in lmc_covariance(1.0, lmc)]
    [[0.606531, 0.303265], [0.303265, 1.213061]]
    """
    lmc, p = _check(lmc)
    out = [[0.0] * p for _ in range(p)]
    for s in lmc:
        r = _rho(abs(float(h)), s)
        for a in range(p):
            for b in range(p):
                out[a][b] += s["B"][a][b] * r
    return out


def lmc_cokriging(z, coords, var, new_coords, lmc, *, target: int = 0, means=None) -> RichResult:
    r"""Cokriging of variable ``target`` from heterotopic multivariate data under a linear model of coregionalization.

    Observation ``i`` is variable ``var[i]`` (0-based) at ``coords[i]`` (any
    dimension: 2-D, 3-D, or space-time as a 4th coordinate). With ``C`` the
    LMC covariance of the observations and ``c_0`` their covariances with
    ``Z_target(s_0)``, ordinary cokriging (default) imposes one unbiasedness
    constraint per variable, ``sum_{var_i = target} lambda_i = 1`` and
    ``sum_{var_i = v} lambda_i = 0`` otherwise (the universal-kriging system
    with indicator design, as ``gstat``); ``means`` (one per variable) gives
    simple cokriging. Returns predictions, cokriging variances and the
    weights (the cokriging weights of every observation).

    References
    ----------
    Wackernagel, H. (2003). *Multivariate Geostatistics*, 3rd edn. Springer, ch. 24-25.
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat package.
    *Computers and Geosciences*, 30, 683-691.

    Examples
    --------
    >>> lmc = [{"model": "Exp", "range": 2.0, "B": [[1.0, 0.6], [0.6, 1.0]]}]
    >>> r = lmc_cokriging([1.0, 2.0, 1.5, 0.5], [(0, 0), (2, 0), (1, 0), (0, 1)], [0, 0, 1, 1], [(1, 1)], lmc)
    >>> round(sum(r.weights[0][:2]), 12), abs(round(sum(r.weights[0][2:]), 12))
    (1.0, 0.0)
    """
    lmc, p = _check(lmc)
    zv = [float(v) for v in z]
    P = [[float(v) for v in (c if isinstance(c, (list, tuple)) else [c])] for c in coords]
    Q = [[float(v) for v in (c if isinstance(c, (list, tuple)) else [c])] for c in new_coords]
    vv = [int(v) for v in var]
    n = len(zv)
    if len(P) != n or len(vv) != n or any(not 0 <= v < p for v in vv) or not 0 <= target < p:
        raise ValueError("z, coords and var must match and variables lie in 0..p-1")

    def dist(a, b):
        s = 0.0
        for x, y in zip(a, b):
            s += (x - y) * (x - y)
        return math.sqrt(s)

    C = [[lmc_covariance(dist(P[i], P[j]), lmc)[vv[i]][vv[j]] for j in range(n)] for i in range(n)]
    Ci = inverse(C)
    c00 = lmc_covariance(0.0, lmc)[target][target]
    pred, varr, W = [], [], []
    for q in Q:
        c0 = [lmc_covariance(dist(P[i], q), lmc)[vv[i]][target] for i in range(n)]
        Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
        if means is not None:
            mu = [float(v) for v in means]
            pk = mu[target] + ssum(Cic0[i] * (zv[i] - mu[vv[i]]) for i in range(n))
            vk = c00 - ssum(c0[i] * Cic0[i] for i in range(n))
            lam = Cic0
        else:
            X = [[1.0 if vv[i] == a else 0.0 for a in range(p)] for i in range(n)]
            used = [a for a in range(p) if any(X[i][a] for i in range(n))]
            if target not in used:
                raise ValueError("ordinary cokriging needs observations of the target variable")
            X = [[X[i][a] for a in used] for i in range(n)]
            x0 = [1.0 if a == target else 0.0 for a in used]
            k = len(used)
            CiX = [[ssum(Ci[i][j] * X[j][a] for j in range(n)) for a in range(k)] for i in range(n)]
            A = [[ssum(X[i][a] * CiX[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
            Ai = inverse(A)
            r = [x0[a] - ssum(X[i][a] * Cic0[i] for i in range(n)) for a in range(k)]
            Air = [ssum(Ai[a][b] * r[b] for b in range(k)) for a in range(k)]
            lam = [Cic0[i] + ssum(CiX[i][a] * Air[a] for a in range(k)) for i in range(n)]
            pk = ssum(lam[i] * zv[i] for i in range(n))
            vk = c00 - ssum(c0[i] * Cic0[i] for i in range(n)) + ssum(r[a] * Air[a] for a in range(k))
        pred.append(pk)
        varr.append(vk)
        W.append(lam)
    return RichResult(payload={"prediction": pred, "variance": varr, "weights": W})


def cheatsheet() -> str:
    return "lmc_covariance / lmc_cokriging -> linear model of coregionalization and cokriging."


# alias kept from the retired placeholder of the same name
cok_weights = lmc_cokriging
