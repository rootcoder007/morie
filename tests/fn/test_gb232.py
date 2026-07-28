"""Tests for gb232 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb232 import gibbons_glivenko_cantelli


def test_gb232_basic():
    rng = np.random.default_rng(0)
    d1 = gibbons_glivenko_cantelli(rng.standard_normal(50))["sup_distance"]
    d2 = gibbons_glivenko_cantelli(rng.standard_normal(5000))["sup_distance"]
    assert d2 < d1


def test_gb232_edge():
    with pytest.raises(ValueError):
        gibbons_glivenko_cantelli([])
