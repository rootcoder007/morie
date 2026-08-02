"""Tests for gb32l2 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb32l2 import gibbons_vandermonde_id1


def test_gb32l2_basic():
    assert gibbons_vandermonde_id1(5, 7)["holds"] is True


def test_gb32l2_edge():
    with pytest.raises(ValueError):
        gibbons_vandermonde_id1(-1, 3)
