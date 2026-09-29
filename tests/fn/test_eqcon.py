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


def test_concentration_index_recomputed():
    """CI = (2 / (n mu)) sum (R_i - 1/2) h_i with fractional SES rank R_i."""
    h = [4.0, 2.5, 6.0, 3.0, 5.5]
    ses = [3.0, 1.0, 5.0, 2.0, 4.0]
    n = 5
    order = sorted(range(n), key=lambda i: ses[i])
    mu = sum(h) / n
    ci = 2 / (n * mu) * sum(((k + 1 - 0.5) / n - 0.5) * h[i] for k, i in enumerate(order))
    assert concentration_index(h, ses).estimate == pytest.approx(ci, rel=1e-13)
