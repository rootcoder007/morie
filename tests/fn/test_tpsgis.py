"""Tests for morie.fn.tpsgis — geo analysis."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.tpsgis import tps_geo_analysis


class TestGeoAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = tps_geo_analysis(rng.normal(43.65, 0.05, 100), rng.normal(-79.38, 0.05, 100))
        assert isinstance(r, DescriptiveResult)
        assert r.extra["centroid_lat"] == pytest.approx(43.65, abs=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tps_geo_analysis([], [])


def test_centroid_spread_and_bbox():
    import math

    lat = [43.65, 43.70, 43.60, 43.72]
    lon = [-79.38, -79.40, -79.30, -79.35]
    ml, mo = sum(lat) / 4, sum(lon) / 4
    r = tps_geo_analysis(lat, lon)
    assert r.extra["centroid_lat"] == pytest.approx(ml, rel=1e-14)
    assert r.extra["std_lon"] == pytest.approx(math.sqrt(sum((v - mo) ** 2 for v in lon) / 4), rel=1e-12)
    assert r.extra["bbox"] == {"min_lat": 43.60, "max_lat": 43.72, "min_lon": -79.40, "max_lon": -79.30}
