"""Tests for gb1323 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1323 import gibbons_are_twosided


def test_gb1323_basic():
    out = gibbons_are_twosided(2.0, 1.0)
    assert out["are_two_sided"] == out["are_one_sided"] == pytest.approx(4.0)


def test_gb1323_edge():
    with pytest.raises(ValueError):
        gibbons_are_twosided(-1.0, 1.0)
