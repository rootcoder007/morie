# morie.fn -- tail3 batch (rootcoder007/morie)
"""GISS surface temperature anomaly.

Source consulted: Hansen, J., Ruedy, R., Glascoe, J. & Sato, M. (1999). GISS
analysis of surface temperature change.  *Journal of Geophysical Research*
104(D24), 30997-31022 (NASA NTRS 19990042165).  A station record is expressed
as an anomaly against its own 1951-1980 base-period climatology,

    anomaly_t = T_t - mean(T over the base period)

and the anomaly at an arbitrary location is the weighted mean of the station
anomalies within 1200 km of it, the weight falling linearly from one at the
point to zero at 1200 km.  Both steps are implemented here; the second is
used only when station distances are supplied.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["giss_anomaly"]


def giss_anomaly(T, years=None, base=(1951, 1980), dist=None, radius=1200.0):
    """Base-period anomaly and its linear trend.

    Parameters
    ----------
    T : array-like
        Temperature record.  A one-dimensional series, or a two-dimensional
        array with one station per row when ``dist`` is supplied.
    years : array-like, optional
        Year of each column.  Defaults to ``0, 1, ..., m-1``, in which case
        the whole record is the base period.
    base : tuple
        Inclusive first and last year of the climatology, 1951-1980 in the
        GISS analysis.
    dist : array-like, optional
        Great-circle distance in km from the target location to each station.
    radius : float
        Influence radius in km, 1200 in the GISS analysis.

    Returns
    -------
    RichResult
        estimate (mean anomaly), anomaly, baseline, trend (per year), nbase,
        n, method.

    References
    ----------
    Hansen, Ruedy, Glascoe & Sato (1999), JGR 104(D24), 30997-31022.
    """
    arr = np.atleast_2d(np.asarray(T, dtype=float))
    nst = int(arr.shape[0])
    m = int(arr.shape[1])
    if years is None:
        yr = np.asarray([float(i) for i in range(m)], dtype=float)
        inbase = [True] * m
    else:
        yr = np.atleast_1d(np.asarray(years, dtype=float)).ravel()
        lo = float(base[0])
        hi = float(base[1])
        inbase = [bool(lo <= float(yr[j]) <= hi) for j in range(m)]
    if not any(inbase):
        inbase = [True] * m
    nbase = int(sum(1 for b in inbase if b))
    anom = []
    baselines = []
    for i in range(nst):
        bm = 0.0
        for j in range(m):
            if inbase[j]:
                bm += float(arr[i, j])
        bm = bm / nbase
        baselines.append(bm)
        anom.append([float(arr[i, j]) - bm for j in range(m)])
    if dist is None:
        w = [1.0 / nst] * nst
    else:
        dd = np.atleast_1d(np.asarray(dist, dtype=float)).ravel()
        raw = [max(0.0, 1.0 - float(dd[i]) / float(radius)) for i in range(nst)]
        tot = sum(raw)
        w = [v / tot for v in raw] if tot > 0.0 else [1.0 / nst] * nst
    series = []
    for j in range(m):
        series.append(sum(w[i] * anom[i][j] for i in range(nst)))
    ser = np.asarray(series, dtype=float)
    xcol = [[1.0, float(yr[j])] for j in range(m)]
    beta = _t3.ols(xcol, ser)
    return RichResult(
        payload={
            "estimate": float(np.mean(ser)),
            "anomaly": ser,
            "baseline": float(sum(w[i] * baselines[i] for i in range(nst))),
            "trend": float(beta[1]),
            "intercept": float(beta[0]),
            "nbase": nbase,
            "nstation": nst,
            "n": m,
            "method": "GISS base-period temperature anomaly (Hansen et al. 1999)",
        }
    )


# CANONICAL TEST
# >>> # anomalies against the whole record average to zero
# >>> r = giss_anomaly([1.0, 2.0, 3.0])
# >>> assert abs(r["estimate"]) < 1e-12
# >>> assert abs(r["baseline"] - 2.0) < 1e-12
# >>> assert abs(r["trend"] - 1.0) < 1e-12


def cheatsheet():
    return "giss(T, years, base): GISS base-period anomaly + linear trend."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
gissanomaly = giss_anomaly
