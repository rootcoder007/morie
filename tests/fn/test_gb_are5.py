"""Tests for gb_are5.gibbons_are_scale_tests."""

import math

import pytest

from morie.fn.gb_are5 import gibbons_are_scale_tests


def test_values():
    r = gibbons_are_scale_tests()
    assert abs(r["are_mood_f"] - 15 / (2 * math.pi**2)) < 1e-15
    assert r["are_klotz_f"] == 1.0
    with pytest.raises(ValueError):
        gibbons_are_scale_tests("logistic")
