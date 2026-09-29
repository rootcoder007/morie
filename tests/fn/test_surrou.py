"""Tests for morie.fn.surrou -- surface roughness."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.surrou import surface_roughness, surrou


class TestSurrou:
    def test_alias(self):
        assert surrou is surface_roughness

    def test_flat_surface(self):
        profile = np.zeros(100)
        r = surface_roughness(profile)
        assert isinstance(r, DescriptiveResult)
        assert r.value == 0.0

    def test_sinusoidal(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        r = surface_roughness(x)
        assert r.value > 0
        assert r.extra["Rz"] > 0


def test_iso_4287_parameters_recomputed():
    import math

    import pytest

    z = [0.5, 1.2, -0.3, 0.8, 2.0, -1.1, 0.4]
    n = 7
    m = sum(z) / n
    c = [v - m for v in z]
    ra = sum(abs(v) for v in c) / n
    rq = math.sqrt(sum(v * v for v in c) / n)
    r = surface_roughness(z, dx=0.5)
    assert r.extra["Ra"] == pytest.approx(ra, rel=1e-13)
    assert r.extra["Rq"] == pytest.approx(rq, rel=1e-13)
    assert r.extra["Rsk"] == pytest.approx(sum(v**3 for v in c) / n / rq**3, rel=1e-12)
    assert r.extra["Rku"] == pytest.approx(sum(v**4 for v in c) / n / rq**4, rel=1e-12)
    assert r.extra["Rz"] == 3.1 and r.extra["profile_length"] == 3.0
