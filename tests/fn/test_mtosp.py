"""Tests for morie.fn.mtosp — speed analysis."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.mtosp import mto_speed_analysis


class TestSpeedAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = mto_speed_analysis(rng.normal(100, 15, 500))
        assert isinstance(r, DescriptiveResult)
        assert r.extra["mean"] == pytest.approx(100, abs=5)

    def test_over_limit(self):
        r = mto_speed_analysis([50, 60, 70, 80, 90], speed_limit=70)
        assert r.extra["pct_over_limit"] == pytest.approx(0.4)


def test_speed_descriptives_recomputed():
    import math

    s = [52.0, 61.0, 47.0, 70.0, 58.0, 66.0]
    m = sum(s) / 6
    xs = sorted(s)
    h = 5 * 0.85
    p85 = xs[4] + (h - 4) * (xs[5] - xs[4])
    r = mto_speed_analysis(s, speed_limit=60)
    assert r.extra["mean"] == pytest.approx(m, rel=1e-14)
    assert r.extra["std"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in s) / 5), rel=1e-13)
    assert r.extra["p85"] == pytest.approx(p85, rel=1e-13)
    assert r.extra["pct_over_limit"] == pytest.approx(3 / 6, rel=1e-15)
