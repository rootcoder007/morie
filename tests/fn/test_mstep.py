"""Tests for morie.fn.mstep."""

import math

import pytest

from morie.fn.mstep import max_step_size


def test_bound():
    x = [math.cos(0.3 * n) + 0.1 * n % 1 for n in range(40)]
    px = math.fsum(t * t for t in x) / 40
    assert abs(max_step_size(x, order=8).value - 2 / (8 * px)) < 1e-15
    with pytest.raises(ValueError):
        max_step_size([0.0, 0.0])
