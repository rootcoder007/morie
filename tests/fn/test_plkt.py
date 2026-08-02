"""Tests for plkt."""

from morie.fn import _array_core as np
import pytest

from morie.fn.plkt import plackett_copula

def test_plkt_basic():
    assert plackett_copula(0.4, 0.6, 1.0)["cdf"] == pytest.approx(0.24)
    assert plackett_copula(0.5, 0.5, 10.0)["cdf"] > 0.25  # positive dependence


def test_plkt_edge():
    with pytest.raises(ValueError):
        plackett_copula(0.5, 0.5, 0.0)
