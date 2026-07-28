"""Tests for gb_med (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_med import gibbons_median_dist


def test_gb_med_basic():
    assert gibbons_median_dist(0.0, 11)["cdf"] == pytest.approx(0.5)


def test_gb_med_edge():
    with pytest.raises(ValueError):
        gibbons_median_dist(0.0, 10)  # even n refused
