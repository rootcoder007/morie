"""Test rr_variability (rrvar)."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.rrvar import rr_variability, rrvar


class TestRrVariability:
    def test_basic(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9, 0.77, 0.83])
        result = rr_variability(rr)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rr_variability"

    def test_sdnn_positive(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9])
        result = rr_variability(rr)
        assert result.extra["sdnn"] > 0

    def test_rmssd_positive(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9])
        result = rr_variability(rr)
        assert result.extra["rmssd"] > 0

    def test_short_rr(self):
        result = rr_variability(np.array([0.8]))
        assert result.value == 0.0

    def test_alias(self):
        assert rrvar is rr_variability


def test_hrv_in_seconds_recomputed():
    import math

    rr = [0.80, 0.86, 0.79, 0.805, 0.90, 0.78]
    m = sum(rr) / 6
    d = [rr[i + 1] - rr[i] for i in range(5)]
    r = rr_variability(rr)
    assert r.extra["sdnn"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in rr) / 5), rel=1e-13)
    assert r.extra["rmssd"] == pytest.approx(math.sqrt(sum(v * v for v in d) / 5), rel=1e-13)
    assert r.extra["pnn50"] == pytest.approx(100 * sum(abs(v) > 0.05 for v in d) / 5, rel=1e-14)
