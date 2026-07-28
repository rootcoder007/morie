"""Tests for gb233 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb233 import gibbons_edf_asymp_normal


def test_gb233_basic():
    out = gibbons_edf_asymp_normal(0.35, 0.3, 100)
    assert out["z"] == pytest.approx(0.05 / np.sqrt(0.21 / 100))


def test_gb233_edge():
    with pytest.raises(ValueError):
        gibbons_edf_asymp_normal(0.5, 0.0, 100)
