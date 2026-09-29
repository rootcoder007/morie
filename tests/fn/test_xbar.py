"""Tests for xbar (sample mean)."""

import pytest

from morie.fn.xbar import xbar


def test_xbar():
    assert xbar([2.0, 4.5, 3.0, 7.5, 1.0]) == pytest.approx(18.0 / 5, rel=1e-15)
    with pytest.raises(ValueError):
        xbar([])
