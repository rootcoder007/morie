# morie.fn -- function file (rootcoder007/morie)
"""Inverse distance weighting (gstat idw conventions), modified Shepard weights, IDW cross-validation and power selection."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["idw_predict", "idw_cv", "idw_power_search", "shepard_predict"]


def _pts(x):
    return [tuple(float(v) for v in (r if isinstance(r, list) else [r])) for r in np.asarray(x, dtype=float).tolist()]


def _aniso(p, angle, ratio):
    if ratio == 1.0:
        return p
    ca, sa = math.cos(angle), math.sin(angle)
    return (ca * p[0] + sa * p[1], (-sa * p[0] + ca * p[1]) / ratio) + tuple(p[2:])


def idw_predict(
    z,
    coords,
    new_coords,
    *,
    power: float = 2.0,
    nmax: int | None = None,
    maxdist: float | None = None,
    block=None,
    angle: float = 0.0,
    ratio: float = 1.0,
) -> RichResult:
    r"""Inverse distance weighted prediction, as ``gstat::idw``.

    ``zhat(s0) = sum w_i z_i / sum w_i`` with ``w_i = d_i^{-power}`` over the
    ``nmax`` nearest observations within ``maxdist`` (``nan`` when none); an
    observation at distance 0 is returned exactly.  ``block`` (offsets, equal
    weights) averages the point predictions over the block discretisation.
    Geometric anisotropy rotates coordinates by ``angle`` (radians, major
    axis) and divides the minor axis by ``ratio`` before measuring distances.
    Also returned: the weighted variance ``sum w_i (z_i - zhat)^2 / sum w_i``
    as a local uncertainty measure.

    References
    ----------
    Shepard, D. (1968). A two-dimensional interpolation function for
    irregularly-spaced data. *Proceedings of the 23rd ACM National
    Conference*, 517-524.
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat
    package. *Computers and Geosciences*, 30(7), 683-691.

    Examples
    --------
    >>> r = idw_predict([1.0, 3.0], [(0, 0), (2, 0)], [(0.5, 0)])
    >>> round(r.prediction[0], 6), round(r.variance[0], 6)
    (1.2, 0.36)
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [_aniso(p, angle, ratio) for p in _pts(coords)]
    Q = [_aniso(q, angle, ratio) for q in _pts(new_coords)]
    offs = [None] if block is None else [tuple(float(v) for v in o) for o in block]

    def point(q):
        d = [math.dist(p, q) for p in P]
        idx = sorted(range(len(P)), key=lambda i: (d[i], i))
        if maxdist is not None:
            idx = [i for i in idx if d[i] <= maxdist]
        if nmax is not None:
            idx = idx[: int(nmax)]
        if not idx:
            return float("nan"), float("nan")
        zero = [i for i in idx if d[i] == 0.0]
        if zero:
            return zv[zero[0]], 0.0
        w = [d[i] ** (-power) for i in idx]
        sw = ssum(w)
        zh = ssum(wi * zv[i] for wi, i in zip(w, idx)) / sw
        return zh, ssum(wi * (zv[i] - zh) ** 2 for wi, i in zip(w, idx)) / sw

    pred, var = [], []
    for q in Q:
        vals = [point(q if o is None else tuple(a + b for a, b in zip(q, _aniso(o, angle, ratio)))) for o in offs]
        pred.append(ssum(v[0] for v in vals) / len(vals))
        var.append(ssum(v[1] for v in vals) / len(vals))
    return RichResult(payload={"prediction": pred, "variance": var})


def idw_cv(
    z, coords, *, power: float = 2.0, nmax: int | None = None, maxdist: float | None = None, folds=None
) -> RichResult:
    r"""Cross-validation of IDW, as ``gstat::krige.cv`` without a variogram model (leave-one-out by default).

    Returns predictions, residuals (observed minus predicted), RMSE, MAE and
    mean error.

    Examples
    --------
    >>> round(idw_cv([1.0, 2.0, 4.0, 3.0], [(0, 0), (1, 0), (2, 0), (3, 0)]).rmse, 6)
    1.155023
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = _pts(coords)
    n = len(zv)
    lab = list(range(n)) if folds is None else [int(v) for v in folds]
    pred = [0.0] * n
    for f in sorted(set(lab)):
        out = [i for i in range(n) if lab[i] == f]
        keep = [i for i in range(n) if lab[i] != f]
        r = idw_predict(
            [zv[i] for i in keep], [P[i] for i in keep], [P[i] for i in out], power=power, nmax=nmax, maxdist=maxdist
        )
        for t, i in enumerate(out):
            pred[i] = r["prediction"][t]
    res = [zv[i] - pred[i] for i in range(n)]
    return RichResult(
        payload={
            "prediction": pred,
            "residual": res,
            "rmse": math.sqrt(ssum(v * v for v in res) / n),
            "mae": ssum(abs(v) for v in res) / n,
            "me": ssum(res) / n,
        }
    )


def idw_power_search(z, coords, powers=None, *, nmax: int | None = None) -> RichResult:
    r"""Leave-one-out RMSE of IDW over a grid of powers and the power minimising it.

    Examples
    --------
    >>> r = idw_power_search([1.0, 2.0, 4.0, 3.0, 5.0], [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [1.0, 2.0, 3.0])
    >>> r.best in (1.0, 2.0, 3.0)
    True
    """
    grid = [0.5 * k for k in range(1, 13)] if powers is None else [float(v) for v in powers]
    rm = [idw_cv(z, coords, power=p, nmax=nmax).rmse for p in grid]
    best = min(range(len(grid)), key=lambda i: (rm[i], i))
    return RichResult(payload={"powers": grid, "rmse": rm, "best": grid[best]})


def shepard_predict(z, coords, new_coords, *, radius: float | None = None, k: int | None = None) -> RichResult:
    r"""Modified Shepard interpolation with Franke and Nielson (1980) weights.

    ``w_i = ((R - d_i)_+ / (R d_i))^2`` with a fixed radius ``R`` or, with
    ``k``, the adaptive radius ``R`` = distance to the ``k``-th nearest
    observation (plus a relative ``1e-12``, so it keeps positive weight);
    ``nan`` when no observation lies within ``R``, exact at data points.

    References
    ----------
    Franke, R. and Nielson, G. (1980). Smooth interpolation of large sets of
    scattered data. *International Journal for Numerical Methods in
    Engineering*, 15(11), 1691-1704.

    Examples
    --------
    >>> round(shepard_predict([1.0, 3.0], [(0, 0), (2, 0)], [(0.5, 0)], radius=3.0).prediction[0], 6)
    1.076923
    """
    if (radius is None) == (k is None):
        raise ValueError("give exactly one of radius and k")
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P, Q = _pts(coords), _pts(new_coords)
    pred = []
    for q in Q:
        d = [math.dist(p, q) for p in P]
        if 0.0 in d:
            pred.append(zv[d.index(0.0)])
            continue
        R = float(radius) if radius is not None else sorted(d)[min(int(k), len(d)) - 1] * (1 + 1e-12)
        w = [((R - di) / (R * di)) ** 2 if di < R else 0.0 for di in d]
        sw = ssum(w)
        pred.append(ssum(wi * zi for wi, zi in zip(w, zv)) / sw if sw > 0 else float("nan"))
    return RichResult(payload={"prediction": pred})


def cheatsheet() -> str:
    return "idw_predict / idw_cv / idw_power_search / shepard_predict -> inverse distance weighting (gstat idw)."
