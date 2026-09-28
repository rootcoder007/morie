# morie.fn -- function file (rootcoder007/morie)
"""Space-time covariance models, empirical space-time variogram and space-time kriging (gstat conventions)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._sci_core import kv

__all__ = ["st_covariance", "st_variogram", "st_kriging_cv"]

_TYPES = ("separable", "productSum", "metric", "sumMetric")


def _comp(d, c):
    """Covariance of a gstat ``vgm`` component: psill rho(d / range) + nugget at d = 0."""
    if c is None:
        return 0.0
    ps, r, nug, mod = (
        float(c.get("psill", 1.0)),
        float(c.get("range", 1.0)),
        float(c.get("nugget", 0.0)),
        c.get("model", "Exp"),
    )
    if d <= 0.0:
        return ps + nug
    h = d / r
    if mod == "Exp":
        rho = math.exp(-h)
    elif mod == "Gau":
        rho = math.exp(-h * h)
    elif mod == "Sph":
        rho = 1.0 - 1.5 * h + 0.5 * h**3 if h < 1.0 else 0.0
    elif mod == "Mat":
        k = float(c.get("kappa", 0.5))
        rho = 2.0 ** (1.0 - k) / math.gamma(k) * h**k * float(kv(k, h))
    elif mod == "Nug":
        rho = 0.0
    else:
        raise ValueError("component model must be Exp, Gau, Sph, Mat or Nug")
    return ps * rho


def st_covariance(ds: float, dt: float, model: dict) -> float:
    r"""Space-time covariance of the gstat ``vgmST`` models at spatial lag ``ds`` and time lag ``dt``.

    - ``separable``: ``sill C_s(h) C_t(u)`` (unit-sill components);
    - ``productSum``: ``C_t(u) + C_s(h) + k C_t(u) C_s(h)`` (De Cesare et al. 2001);
    - ``metric``: ``C_j(sqrt(h^2 + (kappa u)^2))`` with space-time anisotropy ``kappa``;
    - ``sumMetric``: ``C_s(h) + C_t(u) + C_j(sqrt(h^2 + (kappa u)^2))``.

    Components are dicts ``model`` (``Exp``, ``Gau``, ``Sph``, ``Mat`` with
    ``kappa``, ``Nug``), ``psill``, ``range``, ``nugget``, as gstat ``vgm``
    (Graeler, Pebesma and Heuvelink 2016).

    Examples
    --------
    >>> m = {"type": "separable", "sill": 2.0, "space": {"model": "Exp", "range": 1.0}, "time": {"model": "Gau", "range": 2.0}}
    >>> round(st_covariance(1.0, 1.0, m), 6)
    0.57301
    """
    t = model.get("type")
    if t not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}")
    ds, dt = abs(float(ds)), abs(float(dt))
    if t == "separable":
        return float(model.get("sill", 1.0)) * _comp(ds, model["space"]) * _comp(dt, model["time"])
    if t == "productSum":
        cs, ct = _comp(ds, model["space"]), _comp(dt, model["time"])
        return ct + cs + float(model.get("k", 0.0)) * ct * cs
    kap = float(model.get("stAni", 1.0))
    cj = _comp(math.hypot(ds, kap * dt), model["joint"])
    if t == "metric":
        return cj
    return _comp(ds, model["space"]) + _comp(dt, model["time"]) + cj


def st_variogram(z, coords, times, *, tlags, boundaries) -> RichResult:
    r"""Empirical space-time variogram of values on a full space-time grid.

    ``z[t][i]`` is the value at location ``i`` and time ``t``.  For time lag
    ``u`` in ``tlags`` and spatial bin ``(b_k, b_{k+1}]``, preceded by a
    zero-distance bin (the same location at two times, empty for ``u = 0``),
    ``gamma(h_k, u)`` is half the mean squared
    difference ``(z(s_i, t) - z(s_j, t + u))^2`` over all pairs in the bin,
    ``np`` their number and ``dist`` the mean spatial distance, as
    ``gstat::variogramST`` on an ``STFDF`` (pseudo cross-variograms between
    time slices; Graeler, Pebesma and Heuvelink 2016).

    Examples
    --------
    >>> z = [[1.0, 2.0, 4.0], [2.0, 2.5, 3.0]]
    >>> v = st_variogram(z, [(0, 0), (1, 0), (2, 0)], [0, 1], tlags=[0, 1], boundaries=[0, 1.5, 3])
    >>> [round(g, 6) if g == g else None for g in v.gamma]
    [None, 0.6875, 2.5, 0.375, 0.6875, 2.0]
    """
    Z = [[float(v) for v in row] for row in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    T = [float(v) for v in np.asarray(times, dtype=float).tolist()]
    nt, ns = len(Z), len(P)
    if nt != len(T) or any(len(r) != ns for r in Z):
        raise ValueError("z must be times x locations matching coords and times")
    D = [[math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) for j in range(ns)] for i in range(ns)]
    B = [float(v) for v in boundaries]
    out_g, out_n, out_d, out_t, out_s = [], [], [], [], []
    bins = [(None, None)] + [(B[k], B[k + 1]) for k in range(len(B) - 1)]
    for u in tlags:
        u = int(u)
        for lo, hi in bins:
            acc, cnt, dsum = 0.0, 0, 0.0
            for t in range(nt - u):
                for i in range(ns):
                    for j in range(ns):
                        if u == 0 and j <= i:
                            continue
                        d = D[i][j]
                        if (d == 0.0) if lo is None else (lo < d <= hi):
                            acc += 0.5 * (Z[t][i] - Z[t + u][j]) ** 2
                            dsum += d
                            cnt += 1
            out_g.append(acc / cnt if cnt else float("nan"))
            out_n.append(cnt)
            out_d.append(dsum / cnt if cnt else float("nan"))
            out_t.append(u)
            out_s.append(0.0 if lo is None else 0.5 * (lo + hi))
    return RichResult(payload={"gamma": out_g, "np": out_n, "dist": out_d, "timelag": out_t, "spacelag": out_s})


def _st_krige(z, coords, times, new_coords, new_times, model: dict, *, beta: float | None = None) -> RichResult:
    r"""Space-time kriging prediction and variance.

    With the covariance matrix ``C`` of the observations under
    :func:`st_covariance`, ``c_0`` the covariances with the target and
    ``C_00`` the target variance: simple kriging (known mean ``beta``)
    predicts ``beta + c_0' C^{-1} (z - beta)`` with variance
    ``C_00 - c_0' C^{-1} c_0``; ordinary kriging (``beta`` omitted; mean
    estimated by GLS) adds ``(1 - 1' C^{-1} c_0)^2 / (1' C^{-1} 1)`` to the
    variance.  Global neighbourhood, as ``gstat::krigeST`` with ``z ~ 1``.

    :param z: Observed values (n,).
    :param coords: (n, 2) observation locations.
    :param times: (n,) observation times.
    :param new_coords: (m, 2) target locations.
    :param new_times: (m,) target times.
    :param model: Space-time covariance model (:func:`st_covariance`).
    :param beta: Known mean for simple kriging.
    :return: :class:`RichResult` with ``prediction``, ``variance``,
        ``mean`` (the GLS mean for ordinary kriging).

    References
    ----------
    Graeler, B., Pebesma, E. and Heuvelink, G. (2016). Spatio-temporal
    interpolation using gstat. *The R Journal*, 8(1), 204-218.

    Used by :func:`morie.fn.zsstk.st_kriging` and :func:`morie.fn.zsstv.st_kriging_var`.
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    T = [float(v) for v in np.asarray(times, dtype=float).tolist()]
    Q = [(float(a), float(b)) for a, b in np.asarray(new_coords, dtype=float).tolist()]
    U = [float(v) for v in np.asarray(new_times, dtype=float).tolist()]
    n, m = len(zv), len(Q)
    if len(P) != n or len(T) != n or len(U) != m:
        raise ValueError("coords/times must match z and new_coords/new_times each other")

    def cov(p, t, q, s):
        return st_covariance(math.hypot(p[0] - q[0], p[1] - q[1]), t - s, model)

    C = [[cov(P[i], T[i], P[j], T[j]) for j in range(n)] for i in range(n)]
    Ci = [[float(v) for v in r] for r in inverse(C)]
    one = [ssum(r) for r in Ci]
    s1 = ssum(one)
    mu = ssum(one[i] * zv[i] for i in range(n)) / s1 if beta is None else float(beta)
    resid = [v - mu for v in zv]
    w = [ssum(Ci[i][j] * resid[j] for j in range(n)) for i in range(n)]
    pred, var = [], []
    for k in range(m):
        c0 = [cov(P[i], T[i], Q[k], U[k]) for i in range(n)]
        Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
        pred.append(mu + ssum(c0[i] * w[i] for i in range(n)))
        v = cov(Q[k], U[k], Q[k], U[k]) - ssum(c0[i] * Cic0[i] for i in range(n))
        if beta is None:
            v += (1.0 - ssum(Cic0)) ** 2 / s1
        var.append(v)
    return RichResult(payload={"prediction": pred, "variance": var, "mean": mu})


def st_kriging_cv(z, coords, times, model: dict, *, beta: float | None = None) -> RichResult:
    r"""Leave-one-out cross-validation of space-time kriging.

    Each observation is predicted by space-time kriging from all the others
    (as ``gstat::krigeST`` refitted without it); returns the predictions,
    kriging variances, residuals, RMSE and the mean squared standardised
    residual (near 1 for a well-calibrated model).

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> round(st_kriging_cv([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], m).rmse, 6)
    0.593837
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    T = [float(v) for v in np.asarray(times, dtype=float).tolist()]
    n = len(zv)
    pred, var = [], []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        r = _st_krige(
            [zv[j] for j in keep], [P[j] for j in keep], [T[j] for j in keep], [P[i]], [T[i]], model, beta=beta
        )
        pred.append(r["prediction"][0])
        var.append(r["variance"][0])
    res = [zv[i] - pred[i] for i in range(n)]
    return RichResult(
        payload={
            "prediction": pred,
            "variance": var,
            "residuals": res,
            "rmse": math.sqrt(ssum(v * v for v in res) / n),
            "msdr": ssum(res[i] ** 2 / var[i] for i in range(n)) / n,
        }
    )


def cheatsheet() -> str:
    return "st_covariance / st_variogram / st_kriging -> gstat vgmST models, variogramST and krigeST."
