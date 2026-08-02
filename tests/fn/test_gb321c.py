"""Tests for gb321c (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb321c import gibbons_marginal_r1


def test_gb321c_basic():
    # marginal sums to 1 over r1
    total = sum(gibbons_marginal_r1(r, 4, 5)["pmf"] for r in range(1, 5))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_gb321c_edge():
    with pytest.raises(ValueError):
        gibbons_marginal_r1(0, 4, 5)
