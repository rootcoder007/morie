"""Tests for gb321l (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb321l import gibbons_distributing_objects


def test_gb321l_basic():
    assert gibbons_distributing_objects(6, 3)["count"] == 10  # C(5,2)


def test_gb321l_edge():
    with pytest.raises(ValueError):
        gibbons_distributing_objects(3, 5)
