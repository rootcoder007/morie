"""Tests for gb1321 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1321 import gibbons_are_def


def test_gb1321_basic():
    out = gibbons_are_def(1.0, 2.0, n=100)
    assert out["are"] == pytest.approx(0.25)
    assert out["n_star_for_equal_power"] == pytest.approx(25.0)


def test_gb1321_edge():
    with pytest.raises(ValueError):
        gibbons_are_def(0.0, 1.0)
