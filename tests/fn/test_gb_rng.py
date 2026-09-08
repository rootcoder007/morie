"""Tests for gb_rng (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_rng import gibbons_range_dist


def test_gb_rng_basic():
    # range CDF is monotone in w
    assert gibbons_range_dist(2.0, 6)["cdf"] < gibbons_range_dist(4.0, 6)["cdf"]


def test_gb_rng_edge():
    with pytest.raises(ValueError):
        gibbons_range_dist(-1.0, 6)
