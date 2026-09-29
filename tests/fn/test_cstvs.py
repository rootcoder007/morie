"""Tests for morie.fn.cstvs — custody visits."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.cstvs import custody_visits


class TestCustodyVisits:
    def test_returns_descriptive(self):
        vc = np.array([0, 1, 2, 3, 0, 5])
        result = custody_visits(vc)
        assert isinstance(result, DescriptiveResult)

    def test_all_zero(self):
        result = custody_visits(np.array([0, 0, 0]))
        assert result.extra["pct_zero_visits"] == pytest.approx(1.0)


def test_custody_visit_descriptives_recomputed():
    import math

    import pytest

    vc = [0.0, 1.0, 2.0, 3.0, 0.0, 5.0, 4.0]
    m = sum(vc) / 7
    r = custody_visits(vc)
    assert r.value == pytest.approx(m, rel=1e-14)
    assert r.extra["median"] == 2.0
    assert r.extra["std"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in vc) / 7), rel=1e-13)
    assert r.extra["pct_zero_visits"] == pytest.approx(2 / 7, rel=1e-14)
