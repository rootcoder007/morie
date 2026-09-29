"""Tests for morie.fn.eqrii — relative inequality index."""

import pytest

from morie.fn._containers import ESRes
from morie.fn.eqrii import relative_inequality


class TestRelativeInequality:
    def test_basic(self):
        r = relative_inequality([10, 8, 6, 4, 2], [0.1, 0.3, 0.5, 0.7, 0.9])
        assert isinstance(r, ESRes)
        assert r.extra["sii"] < 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            relative_inequality([1, 2], [0.3, 0.7])


def test_rii_is_the_ols_slope_over_the_mean_rate():
    rates = [12.0, 10.5, 9.0, 8.2, 6.1]
    ranks = [0.1, 0.3, 0.5, 0.7, 0.9]
    mx, my = sum(ranks) / 5, sum(rates) / 5
    b = sum((a - mx) * (c - my) for a, c in zip(ranks, rates)) / sum((a - mx) ** 2 for a in ranks)
    r = relative_inequality(rates, ranks)
    assert r.extra["sii"] == pytest.approx(b, rel=1e-12)
    assert r.estimate == pytest.approx(b / my, rel=1e-12)
