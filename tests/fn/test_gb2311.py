"""Tests for gb2311 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb2311 import gibbons_edf_mean_var


def test_gb2311_basic():
    out = gibbons_edf_mean_var(0.3, 40)
    assert out["var"] == pytest.approx(0.3 * 0.7 / 40)


def test_gb2311_edge():
    with pytest.raises(ValueError):
        gibbons_edf_mean_var(1.5, 40)
