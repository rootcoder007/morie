"""Tests for gb_are5 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_are5 import gibbons_are_scale_tests


def test_gb_are5_basic():
    out = gibbons_are_scale_tests()
    assert out["are_mood_f"] == pytest.approx(15 / (2 * np.pi**2))  # PDF-verified
    assert out["are_klotz_f"] == 1.0


def test_gb_are5_edge():
    with pytest.raises(ValueError):
        gibbons_are_scale_tests("uniform")
