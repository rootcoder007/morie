"""Tests for morie.fn.eqcon — concentration index."""

import pytest

from morie.fn._containers import ESRes
from morie.fn.eqcon import concentration_index


class TestConcentration:
    def test_no_inequality(self):
        r = concentration_index([10, 10, 10, 10, 10], [1, 2, 3, 4, 5])
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(0.0, abs=0.01)

    def test_pro_rich(self):
        r = concentration_index([1, 2, 3, 4, 10], [1, 2, 3, 4, 5])
        assert r.estimate > 0
