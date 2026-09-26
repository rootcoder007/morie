"""mrm_tps: grid Moran moments and the Levy tail MLE, against the R arm."""

import math

import pandas as pd

from morie import mrm_tps as T


def test_grid_moran_matches_r_arm():
    k = range(1, 301)
    d = pd.DataFrame(
        {
            "LAT_WGS84": [43.6 + 0.05 * ((v * 7919) % 97) / 97 + 0.01 * math.sin(v) for v in k],
            "LONG_WGS84": [-79.5 + 0.08 * ((v * 104729) % 89) / 89 for v in k],
        }
    )
    r = T.mrm_tps_moran_clustering(d, grid_resolution=6)
    # R arm (randomisation moments, checked there against an explicit W)
    assert (r.morans_I, r.morans_z) == (0.418954, 3.59)


def test_levy_tail_starts_at_x_min():
    lat = [43.6 + v for v in (0, 0.01, 0.03, 0.06, 0.1, 0.2)]
    d = pd.DataFrame(
        {"OCC_DATE": [f"2024-01-{i:02d}" for i in range(1, 7)], "LAT_WGS84": lat, "LONG_WGS84": [-79.4] * 6}
    )
    r = T.mrm_tps_levy_scaling(d, min_step_km=0.5, x_min=2)
    step = [T._haversine_km(a, -79.4, b, -79.4) for a, b in zip(lat[:-1], lat[1:])]
    tail = [s for s in step if s >= 2]
    assert r.n_steps_tail == len(tail)
    assert r.hill_alpha == round(1 + len(tail) / sum(math.log(s / 2) for s in tail), 4)
