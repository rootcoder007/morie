"""Tests for gb_binmw (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_binmw import gibbons_mw_binomial_link


def test_gb_binmw_basic():
    assert gibbons_mw_binomial_link(30.0, 5)["U"] == pytest.approx(15.0)


def test_gb_binmw_edge():
    with pytest.raises(ValueError):
        gibbons_mw_binomial_link(5.0, 5)  # below minimum rank sum
