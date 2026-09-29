# morie.fn -- function file (rootcoder007/morie)
"""Climate indices and climate-change diagnostics: ONI, PDO, QBO, coral degree heating weeks,
cyclone energy (ACE and PDI) and track kinematics, the Hadley-cell edge, Clausius-Clapeyron
scaling, semi-empirical sea level, empirical quantile mapping, BCSD downscaling and the
event-attribution probability ratio."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "oni_index",
    "pdo_index",
    "qbo_index",
    "degree_heating_weeks",
    "cyclone_energy",
    "hurricane_track",
    "hadley_edge",
    "cc_scaling",
    "sea_level_semi_empirical",
    "empirical_quantile_map",
    "bcsd_downscale",
    "probability_ratio",
]

_NAN = float("nan")


def _monthly_anomalies(x, start_month, base):
    n = len(x)
    i0, i1 = (0, n) if base is None else (int(base[0]), int(base[1]))
    clim = []
    for m in range(12):
        vals = [float(x[i]) for i in range(i0, i1) if (start_month - 1 + i) % 12 == m]
        clim.append(ssum(vals) / len(vals) if vals else _NAN)
    anom = [float(x[i]) - clim[(start_month - 1 + i) % 12] for i in range(n)]
    return anom, clim


def _runs(flag, min_run):
    out = [False] * len(flag)
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            if j - i >= min_run:
                for k in range(i, j):
                    out[k] = True
            i = j
        else:
            i += 1
    return out


def oni_index(sst, *, start_month: int = 1, base=None, threshold: float = 0.5, min_run: int = 5) -> RichResult:
    r"""Oceanic Nino Index (NOAA CPC): 3-month running mean of the Nino-3.4 SST anomaly.

    Anomalies are taken from the calendar-month climatology of the months
    ``base = (i0, i1)`` (index range, default the whole series; CPC uses
    centred 30-year base periods). ``oni[i]`` is the mean of the anomalies at
    ``i - 1, i, i + 1`` (NaN at the ends). An El Nino (La Nina) episode is at
    least ``min_run`` consecutive overlapping seasons with
    ``oni >= threshold`` (``<= -threshold``); ``phase`` is 1, -1 or 0.

    References
    ----------
    NOAA Climate Prediction Center, Oceanic Nino Index (ONI) definition.

    Examples
    --------
    >>> r = oni_index([26.0, 27.0, 28.0, 27.0] * 3, start_month=1)
    >>> r.anomalies[:3], r.oni[1]
    ([0.0, 0.0, 0.0], 0.0)
    """
    anom, clim = _monthly_anomalies(sst, start_month, base)
    n = len(anom)
    oni = [_NAN] + [(anom[i - 1] + anom[i] + anom[i + 1]) / 3.0 for i in range(1, n - 1)] + [_NAN]
    warm = _runs([v >= threshold for v in oni], min_run)
    cold = _runs([v <= -threshold for v in oni], min_run)
    phase = [1 if w else (-1 if c else 0) for w, c in zip(warm, cold)]
    return RichResult(payload={"anomalies": anom, "climatology": clim, "oni": oni, "phase": phase})


def _lead_eigen(C):
    w, V = np.linalg.eigh(np.asarray(C, dtype=float))
    w = [float(v) for v in w]
    k = max(range(len(w)), key=lambda i: w[i])
    e = [float(V[r][k]) for r in range(len(w))]
    return w, k, e


def pdo_index(sst, *, start_month: int = 1, weights=None, global_mean=None) -> RichResult:
    r"""Pacific Decadal Oscillation index: leading principal component of North Pacific SST anomalies.

    Following Mantua et al. (1997): monthly anomalies from each point's
    calendar-month climatology, minus the global-mean SST anomaly of that
    month when ``global_mean`` is given, scaled by ``sqrt(weights)`` (area
    weights, e.g. ``cos(latitude)``); the leading eigenvector ``e`` of the
    covariance matrix gives ``PC1 = A e``, standardised to unit variance.
    Sign convention: the loading of largest magnitude (central North
    Pacific) is negative, so positive PDO means a cool central basin.

    References
    ----------
    Mantua, N. J., Hare, S. R., Zhang, Y., Wallace, J. M. and Francis, R. C.
    (1997). A Pacific interdecadal climate oscillation with impacts on salmon
    production. Bull. Amer. Meteor. Soc. 78, 1069-1079. Mantua, N. J. and
    Hare, S. R. (2002). The Pacific Decadal Oscillation. J. Oceanography 58, 35-44.

    Examples
    --------
    >>> x = [[math.sin(i) + 0.1 * j * (i % 7) for j in range(3)] for i in range(36)]
    >>> r = pdo_index(x)
    >>> [round(v, 12) for v in r.loadings]
    [-0.418567239539, -0.564859093601, -0.711150947663]
    """
    T, N = len(sst), len(sst[0])
    cols = [_monthly_anomalies([sst[t][j] for t in range(T)], start_month, None)[0] for j in range(N)]
    A = [[cols[j][t] for j in range(N)] for t in range(T)]
    if global_mean is not None:
        g = _monthly_anomalies(global_mean, start_month, None)[0]
        A = [[A[t][j] - g[t] for j in range(N)] for t in range(T)]
    sw = [1.0] * N if weights is None else [math.sqrt(float(w)) for w in weights]
    A = [[A[t][j] * sw[j] for j in range(N)] for t in range(T)]
    C = [[ssum(A[t][a] * A[t][b] for t in range(T)) / (T - 1) for b in range(N)] for a in range(N)]
    w, k, e = _lead_eigen(C)
    big = max(range(N), key=lambda i: abs(e[i]))
    if e[big] > 0:
        e = [-v for v in e]
    pc = [ssum(A[t][j] * e[j] for j in range(N)) for t in range(T)]
    m = ssum(pc) / T
    sd = math.sqrt(ssum((v - m) ** 2 for v in pc) / (T - 1))
    return RichResult(
        payload={
            "index": [(v - m) / sd for v in pc],
            "pc": pc,
            "loadings": e,
            "eigenvalue": w[k],
            "variance_fraction": w[k] / ssum(w),
        }
    )


def qbo_index(u30, *, start_month: int = 1) -> RichResult:
    r"""Quasi-biennial oscillation index from equatorial 30 hPa zonal wind (Reed et al. 1961).

    The index is the deseasonalised wind (anomaly from the calendar-month
    climatology); ``phase`` is +1 (westerly) or -1 (easterly), ``onsets`` are
    the indices where the index turns from negative to non-negative, and
    ``periods`` the month counts between successive westerly onsets.

    References
    ----------
    Reed, R. J., Campbell, W. J., Rasmussen, L. A. and Rogers, D. G. (1961).
    Evidence of a downward-propagating, annual wind reversal in the equatorial
    stratosphere. J. Geophys. Res. 66, 813-818.

    Examples
    --------
    >>> u = [10.0 * math.sin(2 * math.pi * i / 28) for i in range(84)]
    >>> r = qbo_index(u)
    >>> r.onsets[:2]
    [29, 57]
    """
    anom, clim = _monthly_anomalies(u30, start_month, None)
    phase = [1 if v >= 0 else -1 for v in anom]
    onsets = [i for i in range(1, len(anom)) if anom[i - 1] < 0 <= anom[i]]
    periods = [onsets[i] - onsets[i - 1] for i in range(1, len(onsets))]
    mean_period = ssum(periods) / len(periods) if periods else _NAN
    return RichResult(
        payload={
            "index": anom,
            "climatology": clim,
            "phase": phase,
            "onsets": onsets,
            "periods": periods,
            "mean_period": mean_period,
        }
    )


def degree_heating_weeks(sst, mmm: float, *, days_per_obs: float = 1.0, window_days: float = 84.0) -> RichResult:
    r"""NOAA Coral Reef Watch HotSpots, degree heating weeks and bleaching alert levels.

    ``HotSpot = max(SST - MMM, 0)`` (MMM the maximum monthly mean
    climatology); ``DHW_t = (days_per_obs / 7) * sum HotSpot_j`` over the
    observations in the trailing ``window_days`` (84 days = 12 weeks) with
    ``HotSpot_j >= 1``. Alert levels: 0 no stress, 1 watch (0 < HS < 1),
    2 warning (HS >= 1, DHW < 4), 3 alert level 1 (4 <= DHW < 8), 4 alert level 2 (DHW >= 8).

    References
    ----------
    Liu, G., Strong, A. E. and Skirving, W. (2003). Remote sensing of sea
    surface temperatures during 2002 Barrier Reef coral bleaching. Eos 84(15),
    137-144. NOAA Coral Reef Watch, Degree Heating Week product description.

    Examples
    --------
    >>> r = degree_heating_weeks([29.0, 30.5, 31.0, 30.0], 29.0, days_per_obs=7)
    >>> r.dhw, r.alert
    ([0.0, 1.5, 3.5, 4.5], [0, 2, 2, 3])
    """
    hs = [max(float(v) - mmm, 0.0) for v in sst]
    win = int(round(window_days / days_per_obs))
    dhw = []
    for t in range(len(hs)):
        s = 0.0
        for j in range(max(0, t - win + 1), t + 1):
            if hs[j] >= 1.0:
                s += hs[j]
        dhw.append(s * days_per_obs / 7.0)
    alert = []
    for h, d in zip(hs, dhw):
        if h <= 0.0:
            alert.append(0)
        elif h < 1.0:
            alert.append(1)
        else:
            alert.append(2 if d < 4.0 else (3 if d < 8.0 else 4))
    return RichResult(payload={"hotspot": hs, "dhw": dhw, "alert": alert})


def cyclone_energy(vmax, *, dt_hours: float = 6.0, threshold: float = 35.0) -> RichResult:
    r"""Accumulated cyclone energy and power dissipation index of one storm.

    ``ACE = 1e-4 * sum v^2`` over the fixes with ``v >= threshold`` (knots,
    6-hourly, Bell et al. 2000); ``PDI = sum v^3 * dt`` (Emanuel 2005, with
    ``dt`` in seconds; ``v`` in the caller's units, m/s in Emanuel).

    References
    ----------
    Bell, G. D. et al. (2000). Climate assessment for 1999. Bull. Amer.
    Meteor. Soc. 81, S1-S50. Emanuel, K. (2005). Increasing destructiveness of
    tropical cyclones over the past 30 years. Nature 436, 686-688.

    Examples
    --------
    >>> r = cyclone_energy([30.0, 40.0, 60.0, 50.0])
    >>> round(r.ace, 12), r.pdi
    (0.77, 9331200000.0)
    """
    v = [float(x) for x in vmax]
    ace = 1e-4 * ssum(x * x for x in v if x >= threshold)
    pdi = ssum(x**3 for x in v) * dt_hours * 3600.0
    return RichResult(payload={"ace": ace, "pdi": pdi, "vmax": max(v), "duration_hours": dt_hours * (len(v) - 1)})


def hurricane_track(lat, lon, *, dt_hours: float = 6.0, radius_km: float = 6371.0) -> RichResult:
    r"""Great-circle kinematics of a cyclone track: step lengths, translation speed and heading.

    Step length by the haversine formula, speed = length / ``dt_hours``
    (km/h), heading the initial great-circle bearing (degrees clockwise from
    north); ``recurvature`` is the first fix where the zonal motion turns
    from westward to eastward (None if it never does).

    Examples
    --------
    >>> r = hurricane_track([10.0, 10.0, 11.0], [-40.0, -41.0, -41.0])
    >>> [round(v, 6) for v in r.heading]
    [270.086826, 0.0]
    """
    rad = math.pi / 180.0
    dist, speed, head = [], [], []
    for i in range(len(lat) - 1):
        p1, p2 = lat[i] * rad, lat[i + 1] * rad
        dl = (lon[i + 1] - lon[i]) * rad
        a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        d = 2.0 * radius_km * math.asin(math.sqrt(a))
        dist.append(d)
        speed.append(d / dt_hours)
        b = math.atan2(
            math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        )
        head.append((b / rad) % 360.0)
    rec = None
    for i in range(1, len(lon) - 1):
        if lon[i] - lon[i - 1] < 0 and lon[i + 1] - lon[i] > 0:
            rec = i
            break
    return RichResult(
        payload={
            "distance": dist,
            "speed": speed,
            "heading": head,
            "length": ssum(dist),
            "mean_speed": ssum(speed) / len(speed),
            "recurvature": rec,
        }
    )


def hadley_edge(lat, p, v, *, level: float = 50000.0, radius: float = 6.371e6, g: float = 9.80665) -> RichResult:
    r"""Mean meridional mass streamfunction and the Hadley-cell edges (PSI500 metric).

    ``psi(phi, p) = (2 pi a cos(phi) / g) int_{p_0}^{p} [v] dp'`` by the
    trapezoid rule from the top level ``p[0]`` (Pa, increasing downward;
    ``v[k][j]`` the zonal-mean meridional wind at level ``k`` and latitude
    ``j``, latitudes increasing). At ``level`` (linear in ``p``) the northern
    edge is the first zero crossing poleward of the northern maximum of psi
    and the southern edge the first crossing poleward of the southern
    minimum, located by linear interpolation in latitude.

    References
    ----------
    Davis, S. M. and Rosenlof, K. H. (2012). A multidiagnostic intercomparison
    of tropical-width time series using reanalyses and climate models. J.
    Climate 25, 1061-1078.

    Examples
    --------
    >>> lat = [-40.0, -20.0, 0.0, 20.0, 40.0]
    >>> v = [[1.0, -2.0, 0.0, 2.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    >>> r = hadley_edge(lat, [20000.0, 80000.0], v)
    >>> round(r.edge_north, 12), round(r.edge_south, 12)
    (34.208544286381, -34.208544286381)
    """
    K, J = len(p), len(lat)
    psi = [[0.0] * J for _ in range(K)]
    for j in range(J):
        c = 2.0 * math.pi * radius * math.cos(lat[j] * math.pi / 180.0) / g
        acc = 0.0
        for k in range(1, K):
            acc += 0.5 * (v[k][j] + v[k - 1][j]) * (p[k] - p[k - 1])
            psi[k][j] = c * acc
    k = max(i for i in range(K - 1) if p[i] <= level) if K > 1 else 0
    k = min(k, K - 2)
    f = (level - p[k]) / (p[k + 1] - p[k])
    lev = [psi[k][j] + f * (psi[k + 1][j] - psi[k][j]) for j in range(J)]

    def crossing(js):
        for a, b in zip(js[:-1], js[1:]):
            if lev[a] * lev[b] <= 0 and lev[a] != lev[b]:
                return lat[a] + (lat[b] - lat[a]) * lev[a] / (lev[a] - lev[b])
        return _NAN

    north = [j for j in range(J) if lat[j] > 0]
    south = [j for j in range(J) if lat[j] < 0]
    en = es = _NAN
    if north:
        jm = max(north, key=lambda j: lev[j])
        en = crossing(list(range(jm, J)))
    if south:
        jm = min(south, key=lambda j: lev[j])
        es = crossing(list(range(jm, -1, -1)))
    return RichResult(payload={"psi": psi, "psi_level": lev, "edge_north": en, "edge_south": es})


def _quantile7(x, q):
    s = sorted(float(v) for v in x)
    h = (len(s) - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    w = h - lo
    return (1.0 - w) * s[lo] + w * s[hi] if w > 0 else s[lo]


def _ols1(x, y):
    n = len(x)
    mx, my = ssum(x) / n, ssum(y) / n
    sxx = ssum((a - mx) ** 2 for a in x)
    b = ssum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    return my - b * mx, b


def cc_scaling(temp, precip, *, bin_width: float = 2.0, q: float = 0.99, min_count: int = 10) -> RichResult:
    r"""Clausius-Clapeyron scaling of extreme precipitation with temperature.

    Binning method of Lenderink and van Meijgaard (2008): precipitation is
    binned by temperature (bins of ``bin_width`` from
    ``floor(min T / bin_width) * bin_width``), the ``q`` quantile (type 7)
    is taken in each bin holding at least ``min_count`` values, and
    ``log`` quantile is regressed on bin-centre temperature; the scaling rate
    is ``100 (exp(b) - 1)`` percent per degree. ``cc_rate`` is the
    theoretical rate ``100 d ln e_s / dT`` of the Bolton (1980) saturation
    vapour pressure ``e_s = 6.112 exp(17.67 T / (T + 243.5))`` at the mean
    temperature (about 7 percent per degree).

    References
    ----------
    Lenderink, G. and van Meijgaard, E. (2008). Increase in hourly
    precipitation extremes beyond expectations from temperature changes.
    Nature Geoscience 1, 511-514. Bolton, D. (1980). The computation of
    equivalent potential temperature. Mon. Wea. Rev. 108, 1046-1053.

    Examples
    --------
    >>> t = [float(i % 10) for i in range(40)]
    >>> p = [math.exp(0.07 * v) for v in t]
    >>> r = cc_scaling(t, p, min_count=2)
    >>> round(r.rate, 10)
    7.2508181254
    """
    t = [float(v) for v in temp]
    y = [float(v) for v in precip]
    lo = math.floor(min(t) / bin_width) * bin_width
    nb = int(math.floor((max(t) - lo) / bin_width)) + 1
    centres, quants = [], []
    for b in range(nb):
        vals = [y[i] for i in range(len(t)) if int(math.floor((t[i] - lo) / bin_width)) == b]
        if len(vals) >= min_count:
            qv = _quantile7(vals, q)
            if qv > 0:
                centres.append(lo + (b + 0.5) * bin_width)
                quants.append(qv)
    a, slope = _ols1(centres, [math.log(v) for v in quants])
    tm = ssum(t) / len(t)
    cc = 100.0 * 17.67 * 243.5 / (tm + 243.5) ** 2
    return RichResult(
        payload={
            "rate": 100.0 * (math.exp(slope) - 1.0),
            "slope": slope,
            "intercept": a,
            "bins": centres,
            "quantiles": quants,
            "cc_rate": cc,
        }
    )


def sea_level_semi_empirical(temp, sea_level, *, dt: float = 1.0, future_temp=None) -> RichResult:
    r"""Rahmstorf (2007) semi-empirical sea-level model ``dH/dt = a (T - T0)``.

    The rate ``(H[i+1] - H[i]) / dt`` is regressed on the mid-step
    temperature ``(T[i] + T[i+1]) / 2`` by least squares, giving ``a`` (the
    slope) and ``T0 = -intercept / a``. A future temperature path is
    integrated from the last observation with the same mid-step rule.

    References
    ----------
    Rahmstorf, S. (2007). A semi-empirical approach to projecting future
    sea-level rise. Science 315, 368-370.

    Examples
    --------
    >>> r = sea_level_semi_empirical([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 3.0, 6.0], future_temp=[4.0])
    >>> round(r.a, 12), round(r.T0, 12), [round(v, 12) for v in r.projection]
    (1.0, -0.5, [10.0])
    """
    T = [float(v) for v in temp]
    H = [float(v) for v in sea_level]
    rate = [(H[i + 1] - H[i]) / dt for i in range(len(H) - 1)]
    tm = [(T[i] + T[i + 1]) / 2.0 for i in range(len(T) - 1)]
    b0, a = _ols1(tm, rate)
    T0 = -b0 / a
    proj = []
    if future_temp is not None:
        h, tp = H[-1], T[-1]
        for tf in future_temp:
            h += a * ((tp + float(tf)) / 2.0 - T0) * dt
            proj.append(h)
            tp = float(tf)
    return RichResult(payload={"a": a, "T0": T0, "intercept": b0, "rate": rate, "projection": proj})


def empirical_quantile_map(obs, model_hist, model_future) -> list:
    r"""Empirical quantile mapping ``x' = Q_obs(F_hist(x))`` (bias correction).

    ``F_hist`` is the piecewise-linear empirical CDF of the sorted model
    history on the plotting positions ``(i - 1) / (n - 1)`` (the inverse of
    the type 7 quantile), constant beyond the range; ``Q_obs`` is the type 7
    quantile function of the observations.

    References
    ----------
    Panofsky, H. A. and Brier, G. W. (1968). Some Applications of Statistics
    to Meteorology. Wood, A. W., Leung, L. R., Sridhar, V. and Lettenmaier,
    D. P. (2004). Hydrologic implications of dynamical and statistical
    approaches to downscaling climate model outputs. Climatic Change 62, 189-216.

    Examples
    --------
    >>> empirical_quantile_map([10.0, 20.0, 30.0], [0.0, 1.0, 2.0], [0.5, 1.5, 5.0])
    [15.0, 25.0, 30.0]
    """
    h = sorted(float(v) for v in model_hist)
    n = len(h)
    out = []
    for x in model_future:
        x = float(x)
        if x <= h[0]:
            p = 0.0
        elif x >= h[-1]:
            p = 1.0
        else:
            i = max(k for k in range(n - 1) if h[k] <= x)
            p = (i + (x - h[i]) / (h[i + 1] - h[i])) / (n - 1)
        out.append(_quantile7(obs, p))
    return out


def bcsd_downscale(
    obs_coarse, gcm_hist, gcm_future, coarse_xy, fine_xy, fine_clim, *, multiplicative: bool = False, power: float = 2.0
) -> RichResult:
    r"""Bias correction and spatial disaggregation (BCSD) of Wood et al. (2002, 2004).

    Each coarse cell ``c`` (series ``obs_coarse[c]``, ``gcm_hist[c]``,
    ``gcm_future[c]``) is bias-corrected by empirical quantile mapping; its
    anomaly from the observed climatology (difference, or ratio when
    ``multiplicative``, e.g. precipitation) is interpolated to the fine
    points by inverse-distance weighting with exponent ``power`` (exact at
    coincident points), and applied to the fine-scale climatology
    ``fine_clim``. Returns the fine field (times by fine points).

    References
    ----------
    Wood, A. W., Maurer, E. P., Kumar, A. and Lettenmaier, D. P. (2002).
    Long-range experimental hydrologic forecasting for the eastern United
    States. J. Geophys. Res. 107(D20), 4429. Wood et al. (2004), Climatic Change 62, 189-216.

    Examples
    --------
    >>> r = bcsd_downscale([[1.0, 2.0, 3.0]], [[0.0, 1.0, 2.0]], [[1.0, 2.0]], [(0.0, 0.0)], [(1.0, 0.0)], [5.0])
    >>> r.fine
    [[5.0], [6.0]]
    """
    C = len(obs_coarse)
    bc = [empirical_quantile_map(obs_coarse[c], gcm_hist[c], gcm_future[c]) for c in range(C)]
    clim = [ssum(float(v) for v in obs_coarse[c]) / len(obs_coarse[c]) for c in range(C)]
    Tn = len(bc[0])
    anom = [[bc[c][t] / clim[c] if multiplicative else bc[c][t] - clim[c] for c in range(C)] for t in range(Tn)]
    W = []
    for fx, fy in fine_xy:
        d = [math.hypot(fx - cx, fy - cy) for cx, cy in coarse_xy]
        if min(d) == 0.0:
            j = d.index(0.0)
            w = [1.0 if c == j else 0.0 for c in range(C)]
        else:
            w = [dd ** (-power) for dd in d]
        s = ssum(w)
        W.append([x / s for x in w])
    fine = []
    for t in range(Tn):
        row = []
        for f in range(len(fine_xy)):
            a = ssum(W[f][c] * anom[t][c] for c in range(C))
            row.append(fine_clim[f] * a if multiplicative else fine_clim[f] + a)
        fine.append(row)
    return RichResult(payload={"fine": fine, "bias_corrected": bc, "anomalies": anom})


def probability_ratio(factual, counterfactual, threshold: float) -> RichResult:
    r"""Event-attribution probability ratio and fraction of attributable risk.

    ``p1 = P(X_factual > u)``, ``p0 = P(X_counterfactual > u)`` (empirical
    exceedance frequencies), ``PR = p1 / p0``, ``FAR = 1 - p0 / p1`` and the
    return periods ``1 / p``.

    References
    ----------
    Stott, P. A., Stone, D. A. and Allen, M. R. (2004). Human contribution to
    the European heatwave of 2003. Nature 432, 610-614.

    Examples
    --------
    >>> r = probability_ratio([1, 5, 6, 7], [1, 2, 3, 6], 4.5)
    >>> r.pr, r.far
    (3.0, 0.6666666666666667)
    """
    p1 = ssum(1.0 for v in factual if v > threshold) / len(factual)
    p0 = ssum(1.0 for v in counterfactual if v > threshold) / len(counterfactual)
    pr = p1 / p0 if p0 > 0 else math.inf
    far = 1.0 - p0 / p1 if p1 > 0 else _NAN
    return RichResult(
        payload={
            "p1": p1,
            "p0": p0,
            "pr": pr,
            "far": far,
            "return_period_1": 1.0 / p1 if p1 > 0 else math.inf,
            "return_period_0": 1.0 / p0 if p0 > 0 else math.inf,
        }
    )


def cheatsheet() -> str:
    return (
        "oni_index / pdo_index / qbo_index / degree_heating_weeks / cyclone_energy / hurricane_track / "
        "hadley_edge / cc_scaling / sea_level_semi_empirical / empirical_quantile_map / bcsd_downscale / "
        "probability_ratio -> climate indices and climate-change diagnostics."
    )

# alias kept from the retired placeholder of the same name
bcsd_downscaling = bcsd_downscale

# alias kept from the retired placeholder of the same name
cyclone_intensity = cyclone_energy
