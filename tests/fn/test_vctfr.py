"""Tests for morie.fn.vctfr — fear of crime."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.vctfr import victim_fear


class TestVictimFear:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = victim_fear(rng.integers(1, 6, 200))
        assert isinstance(r, DescriptiveResult)
        assert 1 <= r.value <= 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            victim_fear([])


def test_fear_index_recomputed():
    import math

    s = [1.0, 4.0, 5.0, 3.0, 4.0, 2.0]
    m = sum(s) / 6
    r = victim_fear(s)
    assert r.value == pytest.approx(m, rel=1e-14)
    assert r.extra["std"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in s) / 5), rel=1e-13)
    assert r.extra["pct_high_fear"] == pytest.approx(3 / 6, rel=1e-15)
