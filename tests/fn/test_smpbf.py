"""Tests for morie.fn.smpbf."""

from morie.fn import _array_core as np
from morie.fn.smpbf import smpbf


def test_smpbf_smoke():
    rng = np.random.default_rng(42)
    result = smpbf(lat=rng.uniform(40, 45, size=20), lon=rng.uniform(-80, -75, size=20), radius_km=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.smpbf import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_haversine_buffer_recomputed():
    import math

    import pytest

    lat = [43.65, 43.70, 43.60, 44.10]
    lon = [-79.38, -79.40, -79.30, -79.00]
    c = (43.66, -79.38)

    def hav(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dl = math.radians(b[1] - a[1])
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * 6371.0 * math.asin(math.sqrt(h))

    d = [hav(c, (a, b)) for a, b in zip(lat, lon)]
    r = smpbf(lat, lon, 10.0, center_lat=c[0], center_lon=c[1])
    assert [float(v) for v in r.extra["distances"]] == pytest.approx(d, rel=1e-12)
    assert r.value == sum(v <= 10.0 for v in d)
