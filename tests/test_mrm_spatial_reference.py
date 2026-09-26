"""MRM spatial statistics: LISA (spdep conventions) and the Kulldorff scan."""

import itertools
import math

import pandas as pd

from morie import mrm_kulldorff as K
from morie import mrm_lisa as L


def _polys():
    return pd.DataFrame(
        {
            "lat": [43.6 + 0.01 * math.cos(i) for i in range(1, 9)],
            "lon": [-79.4 + 0.01 * math.sin(1.7 * i) for i in range(1, 9)],
            "x": [5, 9, 2, 7, 3, 8, 1, 6],
        }
    )


def test_lisa_local_i_and_conditional_p_values():
    r = L.mrm_tps_lisa(_polys(), count_col="x", k=2, n_permutations=9999, seed=3)
    # equal to the R arm (spdep::localmoran's I_i)
    assert abs(r.table["I_local"][0] - 0.0106157112527) < 1e-12
    assert abs(r.table["I_local"][2] - (-1.22080679406)) < 1e-10
    assert r.global_moran_I == -0.311
    z, il = list(r.table["z"]), list(r.table["I_local"])
    for j in range(8):
        oth = [z[m] for m in range(8) if m != j]
        ip = [z[j] * (a + b) / 2 for a, b in itertools.combinations(oth, 2)]
        pex = min(sum(v >= il[j] - 1e-12 for v in ip), sum(v <= il[j] + 1e-12 for v in ip)) / len(ip)
        assert abs(r.table["p_value"][j] - pex) < 4 * math.sqrt(pex * (1 - pex) / 9999) + 2 / 10000


def test_kulldorff_expected_count_and_separation():
    n = 400
    rows = []
    for i in range(n):
        hot = i < 80
        day = 1096 + (i * 37) % 700 if hot else (i * 131) % 2500
        lat = 43.66 + 0.003 * math.sin(i) if hot else 43.65 + 0.03 * math.sin(2.3 * i)
        lon = -79.39 + 0.004 * math.cos(i) if hot else -79.38 + 0.04 * math.cos(1.7 * i)
        rows.append(
            {
                "OCC_DATE": f"{(pd.Timestamp('2018-01-01') + pd.Timedelta(days=day)).date()}",
                "LAT_WGS84": lat,
                "LONG_WGS84": lon,
            }
        )
    ev = pd.DataFrame(rows)
    cl = K.mrm_tps_kulldorff_scan(
        ev, radii_km=(1.0, 3.0), window_years=2, n_centers=12, n_permutations=19, n_top_clusters=2, seed=5
    )
    assert len(cl) == 2
    epoch = pd.Timestamp("1970-01-01")
    t = [(pd.Timestamp(s) - epoch).days for s in ev["OCC_DATE"]]
    lats, lons = list(ev["LAT_WGS84"]), list(ev["LONG_WGS84"])
    for c in cl:
        ns = sum(K._haversine_km(c.center_lat, c.center_lon, a, b) <= c.radius_km for a, b in zip(lats, lons))
        ts, te = (pd.Timestamp(c.t_start) - epoch).days, (pd.Timestamp(c.t_end) - epoch).days
        nt = sum(ts <= v < te for v in t)
        assert c.n_expected == round(ns * nt / n, 2)
    assert K._haversine_km(cl[0].center_lat, cl[0].center_lon, cl[1].center_lat, cl[1].center_lon) > cl[0].radius_km
    assert cl[0].log_lrt >= cl[1].log_lrt
