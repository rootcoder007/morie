"""Tests for sampmean (Morin eq 3.54)."""

import pytest

from morie.fn.sampmean import sampmean


def test_sampmean():
    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    r = sampmean(x)
    assert r["mean"] == pytest.approx(sum(x) / 5, rel=1e-15)
    assert r["n"] == 5
