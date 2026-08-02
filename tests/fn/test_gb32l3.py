"""Tests for gb32l3 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb32l3 import gibbons_vandermonde_id2


def test_gb32l3_basic():
    assert gibbons_vandermonde_id2(5, 7)["holds"] is True


def test_gb32l3_edge():
    with pytest.raises(ValueError):
        gibbons_vandermonde_id2(3, 0)
