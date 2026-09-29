"""Tests for morie.fn.swsph: recompute the haversine distances."""

import math

from morie.fn.swsph import swsph

LAT = [43.65, 45.50, 49.28, 46.81, 44.65]
LON = [-79.38, -73.57, -123.12, -71.21, -63.57]


def test_haversine_band():
    r = swsph(LAT, LON, d=800.0)
    for i in range(5):
        for j in range(5):
            p1, p2 = math.radians(LAT[i]), math.radians(LAT[j])
            h = (
                math.sin((p2 - p1) / 2) ** 2
                + math.cos(p1) * math.cos(p2) * math.sin(math.radians(LON[j] - LON[i]) / 2) ** 2
            )
            d = 2 * 6371.0 * math.asin(math.sqrt(h))
            assert abs(r.extra["D"][i][j] - d) < 1e-9
            assert r.extra["W"][i][j] == (1.0 if i != j and d <= 800.0 else 0.0)
