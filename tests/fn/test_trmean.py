"""Tests for trmean (symmetric trimmed mean)."""

import pytest

from morie.fn.trmean import trmean


def test_trimmed_mean_cuts_floor_trim_n_each_side():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    r = trmean(x, trim=0.2)
    assert r["value"] == pytest.approx(sum(sorted(x)[2:8]) / 6, rel=1e-14)
    assert r["arithmetic_mean"] == pytest.approx(sum(x) / 10, rel=1e-14)
    assert r["n_trimmed"] == 4
