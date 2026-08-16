"""HadCRUT5 blended near-surface temperature anomaly, with uncertainty.

A global temperature record is two measurement systems stapled together.
Over land it is air temperature from weather stations; over the ocean it
is water temperature from ships and buoys. They are different physical
quantities measured by different instruments, and the record is the
weighted average of the two in each 5 degree cell.

The weights are where the judgement lives, and HadCRUT5 makes three
decisions worth spelling out, because each of them changes the answer:

  * the weight is the AREAL FRACTION of land and sea in the cell, from
    the OSTIA land mask;
  * land air temperature is given a MINIMUM weighting of 25%, so that a
    single island station in an otherwise oceanic cell is not averaged
    into nothing. The floor applies only where the land data set
    actually reports the cell -- in a cell with no land observation the
    ocean takes the whole weight, and no floor is invented for a
    station that is not there;
  * water under sea ice is not ocean for this purpose. Ice concentration
    above 15% on the native grid counts the area as ice covered, and
    ice-covered water is treated as LAND when the weights are derived.

Where only one of the two data sets reports a cell, that one gets the
whole weight.

Averaging up from cells is the second place a choice hides. A cell at
70 degrees north covers a fraction of the area of a cell on the equator,
so cells enter an area mean weighted by the cosine of their latitude.
HadCRUT5's headline global series is not that: it is the UNWEIGHTED mean
of the two hemispheric means, which is a deliberate coverage decision --
it stops the better-observed hemisphere from dominating the global
figure. Both are here, and so is the land-record convention of weighting
the northern hemisphere twice, because the routes disagree exactly when
coverage is asymmetric, which is most of the nineteenth century.

Uncertainty is carried as three separate things, because they do not
combine the same way:

  "uncorrelated"  measurement and sampling error in a cell, independent
                  between cells, so it shrinks under averaging;
  "correlated"    bias adjustment error, which does NOT shrink, and is
                  carried by an ensemble of realisations of the whole
                  field;
  "coverage"      the error from averaging the observed cells instead of
                  the globe. This one cannot be computed from the
                  observations alone -- it is estimated by taking a
                  COMPLETE field, masking it to the coverage actually
                  achieved, and comparing the two averages. Pass such a
                  field (or several) as `reference` and that is exactly
                  what happens here. With no reference field the term is
                  reported as None rather than guessed.

References
  Morice, C.P., Kennedy, J.J., Rayner, N.A., Winn, J.P., Hogan, E.,
    Killick, R.E., Dunn, R.J.H., Osborn, T.J., Jones, P.D. and Simpson,
    I.R. (2021) "An updated assessment of near-surface temperature
    change from 1850: the HadCRUT5 data set." Journal of Geophysical
    Research: Atmospheres 126(3), e2019JD032361.
    doi:10.1029/2019JD032361. The blending weights, the 25% land floor,
    the sea-ice rule and the hemispheric global mean.
  Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R.,
    Winn, J.P., Hogan, E. and Harris, I.C. (2021) "Land surface air
    temperature variations across the globe updated to 2019: the
    CRUTEM5 data set." Journal of Geophysical Research: Atmospheres
    126(2), e2019JD032352. The land component.
  Kennedy, J.J., Rayner, N.A., Atkinson, C.P. and Killick, R.E. (2019)
    "An ensemble data set of sea surface temperature change from 1850:
    the Met Office Hadley Centre HadSST.4.0.0.0 data set." Journal of
    Geophysical Research: Atmospheres 124(14), 7719-7763. The ocean
    component and the correlated-error ensemble.
  Donlon, C.J., Martin, M., Stark, J., Roberts-Jones, J., Fiedler, E.
    and Wimmer, W. (2012) "The Operational Sea Surface Temperature and
    Sea Ice Analysis (OSTIA) system." Remote Sensing of Environment 116,
    140-158. The land mask the areal fractions come from.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["hadcrut", "blend_weights", "blend_grid", "area_mean",
           "coverage_error", "WEIGHT_RULES", "MEAN_ROUTES", "INTERVALS",
           "LAND_FLOOR", "ICE_THRESHOLD", "cheatsheet"]

WEIGHT_RULES = ("hadcrut5", "area", "land_only", "sst_only")
MEAN_ROUTES = ("hemispheric", "area", "land_ratio")
INTERVALS = ("normal", "ensemble")

# The minimum weight land air temperature receives in a cell the land
# data set reports, so an island station is not averaged away.
LAND_FLOOR = 0.25
# Ice concentration above this counts the area as ice covered, and
# ice-covered water is weighted as land.
ICE_THRESHOLD = 0.15


def blend_weights(land_fraction, sea_ice=0.0, has_land=True, has_sst=True,
                  rule="hadcrut5"):
    """Land and ocean weights for one cell.

    Returns (w_land, w_sst), which sum to one whenever the cell has any
    observation at all and to zero when it has none.

    The floor is applied only when the land data set reports the cell.
    Applying it to an unobserved cell would put weight on a station that
    does not exist, which is the one thing the rule is not for.
    """
    if rule not in WEIGHT_RULES:
        raise ValueError("rule must be one of %r" % (WEIGHT_RULES,))
    lf = float(land_fraction)
    if lf < 0.0 or lf > 1.0:
        raise ValueError("land_fraction must lie in [0, 1]")
    ice = float(sea_ice)
    if ice < 0.0 or ice > 1.0:
        raise ValueError("sea_ice must lie in [0, 1]")
    if rule == "land_only":
        return (1.0, 0.0) if has_land else (0.0, 0.0)
    if rule == "sst_only":
        return (0.0, 1.0) if has_sst else (0.0, 0.0)
    # Ice-covered water is land for the purpose of the weights. Below
    # the concentration threshold the ice is ignored entirely rather
    # than scaled down -- the rule is a threshold, not a ramp.
    frac_ice = ice if ice >= ICE_THRESHOLD else 0.0
    eff = lf + (1.0 - lf) * frac_ice
    if not has_land and not has_sst:
        return (0.0, 0.0)
    if not has_sst:
        return (1.0, 0.0)
    if not has_land:
        return (0.0, 1.0)
    if rule == "hadcrut5" and eff < LAND_FLOOR:
        eff = LAND_FLOOR
    return (eff, 1.0 - eff)


def _cell_lat(i, n_lat):
    """Centre latitude of row i of an n_lat band grid, south to north."""
    band = 180.0 / n_lat
    return -90.0 + (i + 0.5) * band


def blend_grid(T, sst, land_fraction, sea_ice=None, rule="hadcrut5",
               T_var=None, sst_var=None):
    """Blend a land grid and an SST grid cell by cell.

    Missing values are None in either grid. Returns the blended anomaly
    grid, the blended variance grid (None where the inputs carry no
    variance), the land weights and the observation mask.
    """
    n_lat = len(T)
    n_lon = len(T[0])
    anom = []
    var = []
    wl = []
    seen = []
    for i in range(n_lat):
        ra, rv, rw, rs = [], [], [], []
        for j in range(n_lon):
            tl = T[i][j]
            ts = sst[i][j]
            hl = tl is not None
            hs = ts is not None
            ice = 0.0 if sea_ice is None else float(sea_ice[i][j])
            a, b = blend_weights(land_fraction[i][j], ice, hl, hs, rule)
            if a + b <= 0.0:
                ra.append(None)
                rv.append(None)
                rw.append(0.0)
                rs.append(False)
                continue
            v = 0.0
            if a > 0.0:
                v += a * float(tl)
            if b > 0.0:
                v += b * float(ts)
            ra.append(v)
            if T_var is None and sst_var is None:
                rv.append(None)
            else:
                # Independent sources, so the variances add through the
                # SQUARED weights -- the usual trap is to add them
                # through the weights themselves, which understates a
                # near-even blend and overstates a lopsided one.
                q = 0.0
                if a > 0.0 and T_var is not None and T_var[i][j] is not None:
                    q += a * a * float(T_var[i][j])
                if b > 0.0 and sst_var is not None and sst_var[i][j] is not None:
                    q += b * b * float(sst_var[i][j])
                rv.append(q)
            rw.append(a)
            rs.append(True)
        anom.append(ra)
        var.append(rv)
        wl.append(rw)
        seen.append(rs)
    return anom, var, wl, seen


def _band_weight(i, n_lat):
    """Area weight of a latitude band: the cosine of its centre."""
    return math.cos(_cell_lat(i, n_lat) * math.pi / 180.0)


def _region_mean(grid, rows, var=None):
    """Cosine-weighted mean over the given rows, and its variance.

    Returns (mean, variance, weight, n_cells). The mean is None when the
    region holds no observation, which is a real state in 1850 and must
    not silently become zero.
    """
    n_lat = len(grid)
    num = []
    den = []
    qnum = []
    n = 0
    for i in rows:
        w = _band_weight(i, n_lat)
        for j in range(len(grid[i])):
            if grid[i][j] is None:
                continue
            num.append(w * grid[i][j])
            den.append(w)
            n += 1
            if var is not None and var[i][j] is not None:
                qnum.append(w * w * var[i][j])
    if not den:
        return None, None, 0.0, 0
    d = _w.csum(den)
    m = _w.csum(num) / d
    q = _w.csum(qnum) / (d * d) if qnum else None
    return m, q, d, n


def area_mean(grid, route="hemispheric", var=None):
    """Average a grid up to a global figure.

    "area"         one cosine-weighted mean over every observed cell.
    "hemispheric"  the mean of the two hemispheric means, which is
                   HadCRUT5's headline convention: it stops the
                   better-observed hemisphere from carrying the global
                   number.
    "land_ratio"   the land-record convention, two parts north to one
                   part south, in the ratio of the hemispheres' land
                   areas.
    """
    if route not in MEAN_ROUTES:
        raise ValueError("route must be one of %r" % (MEAN_ROUTES,))
    n_lat = len(grid)
    south = [i for i in range(n_lat) if _cell_lat(i, n_lat) < 0.0]
    north = [i for i in range(n_lat) if _cell_lat(i, n_lat) >= 0.0]
    sm, sv, sw, sn = _region_mean(grid, south, var)
    nm, nv, nw, nn = _region_mean(grid, north, var)
    if route == "area":
        m, v, w, n = _region_mean(grid, list(range(n_lat)), var)
        out = {"mean": m, "var": v, "weight": w, "n_cells": n}
    else:
        a, b = (0.5, 0.5) if route == "hemispheric" else (2.0 / 3.0, 1.0 / 3.0)
        if nm is None and sm is None:
            out = {"mean": None, "var": None, "weight": 0.0, "n_cells": 0}
        elif nm is None:
            out = {"mean": sm, "var": sv, "weight": sw, "n_cells": sn}
        elif sm is None:
            out = {"mean": nm, "var": nv, "weight": nw, "n_cells": nn}
        else:
            out = {"mean": a * nm + b * sm,
                   "var": (None if (nv is None or sv is None)
                           else a * a * nv + b * b * sv),
                   "weight": nw + sw, "n_cells": nn + sn}
    out["north"] = nm
    out["south"] = sm
    out["n_north"] = nn
    out["n_south"] = sn
    out["route"] = route
    return out


def coverage_error(reference, seen, route="hemispheric"):
    """The coverage error of one complete field under one coverage mask.

    The complete field is averaged twice: over everything, and over only
    the cells the observations actually reach. The difference is one
    realisation of the coverage error -- not a bound on it, a draw from
    it -- and the root mean square over several reference fields is the
    coverage uncertainty.

    This is the only honest way to get the term. It cannot be derived
    from the observations, because the observations are precisely what
    is missing where it matters.
    """
    full = area_mean(reference, route)["mean"]
    masked = [[reference[i][j] if seen[i][j] else None
               for j in range(len(reference[i]))]
              for i in range(len(reference))]
    part = area_mean(masked, route)["mean"]
    if full is None or part is None:
        return None
    return part - full


def hadcrut(T, sst, land_fraction=None, sea_ice=None, rule="hadcrut5",
            route="hemispheric", interval="normal", level=0.95,
            T_var=None, sst_var=None, ensemble=None, reference=None):
    """Blend land and ocean anomaly grids and average them up.

    Parameters
    ----------
    T : sequence of sequences
        Land air temperature anomalies on a regular latitude/longitude
        grid running south to north, None where unobserved.
    sst : sequence of sequences
        Sea-surface temperature anomalies on the same grid.
    land_fraction : sequence of sequences or None
        Areal land fraction per cell. All ocean when omitted.
    sea_ice : sequence of sequences or None
        Sea-ice concentration per cell. Concentration at or above
        ICE_THRESHOLD makes the water count as land for the weights.
    rule : str
        A member of WEIGHT_RULES.
    route : str
        A member of MEAN_ROUTES.
    interval : str
        "normal" builds the interval from the combined standard error;
        "ensemble" takes it from the spread of `ensemble`, which is the
        only route that carries the correlated bias term properly.
    level : float
        Interval coverage.
    T_var, sst_var : sequence of sequences or None
        Per-cell uncorrelated error variances.
    ensemble : sequence of grids or None
        Realisations of the blended field. Their spread is the
        correlated component.
    reference : sequence of grids or None
        Complete fields used to estimate the coverage term.

    Returns
    -------
    RichResult
        The blended grid, the global and hemispheric means, the three
        uncertainty components and the combined interval, and the
        coverage actually achieved.

    References
    ----------
    Morice et al. (2021) JGR Atmospheres 126(3), e2019JD032361.
    """
    if rule not in WEIGHT_RULES:
        raise ValueError("rule must be one of %r" % (WEIGHT_RULES,))
    if route not in MEAN_ROUTES:
        raise ValueError("route must be one of %r" % (MEAN_ROUTES,))
    if interval not in INTERVALS:
        raise ValueError("interval must be one of %r" % (INTERVALS,))
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    n_lat = len(T)
    if n_lat < 2:
        raise ValueError("need at least two latitude bands")
    n_lon = len(T[0])
    for g, nm in ((T, "T"), (sst, "sst")):
        if len(g) != n_lat or any(len(r) != n_lon for r in g):
            raise ValueError("%s must be a rectangular grid matching T" % nm)
    if land_fraction is None:
        land_fraction = [[0.0] * n_lon for _ in range(n_lat)]

    anom, var, wl, seen = blend_grid(T, sst, land_fraction, sea_ice, rule,
                                     T_var, sst_var)
    agg = area_mean(anom, route, var)
    est = agg["mean"]

    n_obs = sum(1 for i in range(n_lat) for j in range(n_lon) if seen[i][j])
    # Coverage as a fraction of AREA, not of cells: a missing polar cell
    # is much less of a gap than a missing tropical one.
    tot = _w.csum(_band_weight(i, n_lat) for i in range(n_lat)
                  for _ in range(n_lon))
    got = _w.csum(_band_weight(i, n_lat) for i in range(n_lat)
                  for j in range(n_lon) if seen[i][j])
    se_unc = math.sqrt(agg["var"]) if agg.get("var") is not None else None

    se_cor = None
    members = None
    if ensemble is not None:
        members = []
        for g in ensemble:
            members.append(area_mean(g, route)["mean"])
        members = [m for m in members if m is not None]
        if len(members) > 1:
            mm = _w.csum(members) / len(members)
            se_cor = math.sqrt(_w.csum((m - mm) * (m - mm) for m in members)
                               / (len(members) - 1))

    se_cov = None
    cov_draws = None
    if reference is not None:
        cov_draws = [coverage_error(r, seen, route) for r in reference]
        cov_draws = [c for c in cov_draws if c is not None]
        if cov_draws:
            se_cov = math.sqrt(_w.csum(c * c for c in cov_draws)
                               / len(cov_draws))

    parts = [s for s in (se_unc, se_cor, se_cov) if s is not None]
    se = math.sqrt(_w.csum(p * p for p in parts)) if parts else None

    lo = hi = None
    if est is not None:
        if interval == "ensemble" and members and len(members) > 1:
            srt = sorted(members)
            # The empirical quantile at the nearest rank, which is the
            # convention that needs no interpolation and therefore
            # cannot disagree between two languages' quantile types.
            for tail, dest in ((0.5 * (1.0 - level), "lo"),
                               (1.0 - 0.5 * (1.0 - level), "hi")):
                k = int(math.ceil(tail * len(srt))) - 1
                if k < 0:
                    k = 0
                if k >= len(srt):
                    k = len(srt) - 1
                if dest == "lo":
                    lo = srt[k]
                else:
                    hi = srt[k]
        elif se is not None:
            z = _w.nppf(1.0 - 0.5 * (1.0 - level))
            lo = est - z * se
            hi = est + z * se

    return RichResult(payload={
        "anomaly": anom,
        "variance": var,
        "land_weight": wl,
        "observed": seen,
        "estimate": est,
        "se": se,
        "se_uncorrelated": se_unc,
        "se_correlated": se_cor,
        "se_coverage": se_cov,
        "ci_lower": lo,
        "ci_upper": hi,
        "level": float(level),
        "north": agg["north"],
        "south": agg["south"],
        "n_north": agg["n_north"],
        "n_south": agg["n_south"],
        "n_observed": n_obs,
        "n_cells": n_lat * n_lon,
        "coverage": got / tot if tot > 0.0 else float("nan"),
        "coverage_draws": cov_draws,
        "ensemble_means": members,
        "rule": rule,
        "route": route,
        "interval": interval,
        "method": "HadCRUT5 blended anomaly",
    })


def cheatsheet():
    return ("hadcrut: HadCRUT5 blended land/SST anomaly. rules "
            + ", ".join(WEIGHT_RULES) + "; routes " + ", ".join(MEAN_ROUTES)
            + "; intervals " + ", ".join(INTERVALS))
