"""Tests for eudst."""

import numpy as np
import pytest

from morie.fn.eudst import euclidean_utility


def test_eudst_basic():
    out = euclidean_utility([0.0, 0.0], [3.0, 4.0])
    assert out["utility"] == pytest.approx(-25.0)
    assert out["distance"] == pytest.approx(5.0)


def test_eudst_edge():
    with pytest.raises(ValueError):
        euclidean_utility([0.0], [1.0, 2.0])  # dimension mismatch
