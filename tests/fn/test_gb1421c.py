"""Tests for gb1421c (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1421c import gibbons_contingency_coeff


def test_gb1421c_basic():
    out = gibbons_contingency_coeff([[18, 7], [6, 19]])
    assert 0 < out["C"] < out["C_max"] < 1


def test_gb1421c_edge():
    with pytest.raises(ValueError):
        gibbons_contingency_coeff([[1.0]])
