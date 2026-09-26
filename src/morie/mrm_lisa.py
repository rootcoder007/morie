# SPDX-License-Identifier: AGPL-3.0-or-later
"""LISA (Local Indicators of Spatial Association) + per-year polygon
Moran's I on TPS-style geographic crime data.

Two callables:

1. mrm_tps_lisa(data, year=...) -- Local Moran's I per polygon centroid,
   with 999-permutation Monte-Carlo significance. Returns per-polygon
   I_i, lag z-score, quadrant (HH/HL/LH/LL), and p-value. Computes
   the global Moran's I as a by-product.

2. mrm_tps_polygon_moran_per_year(data, year_cols=...) -- convenience
   wrapper that loops mrm_tps_lisa over a set of per-year count
   columns (e.g. ASSAULT_2014 ... ASSAULT_2024) and returns the
   global Moran's I time series. Used by the empirical paper §7.11
   to document the declining spatial concentration of assault
   2014-2024.

Both functions take a GeoJSON-like polygon frame (or a DataFrame
with lat/lon centroid columns + per-feature counts) and compute
k-nearest-neighbour spatial weights (default k=6).

References:
    Anselin, L. (1995). Local indicators of spatial association -- LISA.
        Geographical Analysis, 27(2), 93-115.
    Anselin, L. (2010). Thirty years of spatial econometrics.
        Papers in Regional Science, 89(1), 3-25.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = [
    "LISAResult",
    "mrm_tps_lisa",
    "mrm_tps_polygon_moran_per_year",
]


@dataclass
class LISAResult:
    """Output of mrm_tps_lisa."""

    n_polygons: int
    global_moran_I: float
    permutations: int
    knn_k: int
    table: pd.DataFrame  # per-polygon: id, lat, lon, x, z, lag_z, I_local, quadrant, p_value
    quadrants_all: dict
    quadrants_significant_p05: dict
    n_significant_p05: int


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _knn_weights(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    n = lat.size
    W = np.zeros((n, n))
    for i in range(n):
        d = _haversine_km(lat[i], lon[i], lat, lon)
        nn = np.argsort(d)[1 : k + 1]
        W[i, nn] = 1.0 / k
    return W


def mrm_tps_lisa(
    data: pd.DataFrame,
    *,
    count_col: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str | None = None,
    k: int = 6,
    n_permutations: int = 999,
    seed: int = 42,
) -> LISAResult:
    """Local Moran's I + quadrant + significance for a polygon surface.

    Args:
        data: DataFrame with one row per polygon. Must have lat/lon
            centroid columns + a count_col with per-polygon counts.
        count_col: per-polygon count column (e.g. "ASSAULT_2024").
        lat_col, lon_col: WGS84 centroid columns.
        id_col: optional polygon-id column (passed through to output).
        k: k-NN spatial-weights neighbourhood (default 6).
        n_permutations: MC permutations for significance. 999 is the
            spatial-statistics convention. Local p-values use conditional
            randomization (Anselin 1995): z_i stays fixed and its
            neighbours are drawn from the other n - 1 values; folded
            pseudo p = (min(#>=, #<=) + 1) / (R + 1), as GeoDa and
            spdep::localmoran_perm.
        seed: RNG seed for reproducibility.

    Returns:
        LISAResult with per-polygon table and quadrant summaries.
    """
    rng = np.random.default_rng(seed)
    df = data[[count_col, lat_col, lon_col] + ([id_col] if id_col else [])].copy()
    df = df.dropna(subset=[count_col, lat_col, lon_col])
    n = len(df)
    if n < 5:
        raise ValueError(f"need >= 5 polygons; got {n}")

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    x = df[count_col].to_numpy(dtype=float)

    W = _knn_weights(lat, lon, k)
    z = (x - x.mean()) / x.std(ddof=0)
    lag = W @ z
    I_local = z * lag
    I_global = float(I_local.sum() / (z**2).sum())

    # Quadrants
    # labels as a plain list: the array shim holds numbers only
    quad = [("H" if zi > 0 else "L") + ("H" if li > 0 else "L") for zi, li in zip(z, lag)]

    # Significance via MC permutation (z fixed, lag permuted)
    # conditional randomization (Anselin 1995; GeoDa, spdep::localmoran_perm):
    # z_i stays at i and its neighbours are drawn from the other n - 1
    # values; folded pseudo p = (min(#>=, #<=) + 1) / (R + 1)
    p_local = np.zeros(n)
    for i in range(n):
        nb = np.nonzero(W[i])[0]
        w = W[i, nb]
        others = np.delete(z, i)
        ge = le = 0
        # ties (the observed neighbour set, repeated values) count both ways
        tol = 1e-10 * max(1.0, abs(float(I_local[i])))
        for _ in range(n_permutations):
            Ip = z[i] * float(w @ others[rng.choice(n - 1, size=nb.size, replace=False)])
            ge += Ip >= I_local[i] - tol
            le += Ip <= I_local[i] + tol
        p_local[i] = (min(ge, le) + 1) / (n_permutations + 1)

    out = pd.DataFrame(
        {
            "id": df[id_col].to_numpy() if id_col else range(n),
            "lat": lat,
            "lon": lon,
            "x": x,
            "z": z,
            "lag_z": lag,
            "I_local": I_local,
            "quadrant": quad,
            "p_value": p_local,
            "significant_p05": p_local <= 0.05,
        }
    )

    quads_all = {q: sum(v == q for v in quad) for q in ("HH", "HL", "LH", "LL")}
    quads_sig = {q: sum(v == q and pv <= 0.05 for v, pv in zip(quad, p_local)) for q in ("HH", "HL", "LH", "LL")}

    return LISAResult(
        n_polygons=int(n),
        global_moran_I=round(I_global, 4),
        permutations=int(n_permutations),
        knn_k=int(k),
        table=out,
        quadrants_all=quads_all,
        quadrants_significant_p05=quads_sig,
        n_significant_p05=int(sum(quads_sig.values())),
    )


def mrm_tps_polygon_moran_per_year(
    data: pd.DataFrame,
    *,
    year_cols: Sequence[str],
    lat_col: str = "lat",
    lon_col: str = "lon",
    k: int = 6,
    n_permutations: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-year global Moran's I time series across a polygon surface.

    Args:
        data: DataFrame with one row per polygon + one count column
            per year.
        year_cols: list of per-year count columns (in year order).
        lat_col, lon_col: WGS84 centroid columns.
        k: k-NN spatial-weights neighbourhood.
        n_permutations: MC permutations for each year's significance.
        seed: RNG seed.

    Returns:
        DataFrame with one row per year: year, n_events, moran_I,
        global_p_value (one-sided permutation test of positive
        autocorrelation, as spdep::moran.mc).
    """
    rows = []
    for c in year_cols:
        # Try to parse year out of the column name
        import re

        m = re.search(r"\d{4}", c)
        year = int(m.group(0)) if m else c

        # the polygons mrm_tps_lisa uses: count and centroid all present
        d = data[[c, lat_col, lon_col]].dropna()
        if len(d) < 5:
            continue

        # Global p via permutation of the whole z surface
        rng = np.random.default_rng(seed)
        x = d[c].to_numpy(dtype=float)
        z = (x - x.mean()) / x.std(ddof=0)
        W = _knn_weights(d[lat_col].to_numpy(dtype=float), d[lon_col].to_numpy(dtype=float), k)
        I_obs = float((z * (W @ z)).sum() / (z**2).sum())
        # one-sided test of positive autocorrelation, as spdep::moran.mc
        # (E[I] = -1/(n-1), so |I| >= |I_obs| is not a two-sided test)
        ge = 0
        for _ in range(n_permutations):
            zp = rng.permutation(z)
            ge += (zp * (W @ zp)).sum() / (zp**2).sum() >= I_obs
        p_global = (ge + 1) / (n_permutations + 1)

        rows.append(
            {
                "year": year,
                "n_events": int(x.sum()),
                "moran_I": round(I_obs, 4),
                "global_p_value": round(p_global, 4),
            }
        )

    return pd.DataFrame(rows)
