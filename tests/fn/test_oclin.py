"""Tests for oclin."""

import numpy as np
import pytest

from morie.fn.oclin import oc_cutting_line


def test_oclin_basic():
    x = np.array([-1.0, -0.6, 0.1, 0.4, 0.9])
    v = np.array([1, 1, 1, 0, 0], dtype=float)
    out = oc_cutting_line(x, v)
    assert out["errors"] == 0
    assert 0.1 < out["cutpoint"] < 0.4


def test_oclin_edge():
    with pytest.raises(ValueError):
        oc_cutting_line(np.array([0.0, 1.0]), np.array([1.0, 1.0]))  # one class
    with pytest.raises(ValueError):
        oc_cutting_line(np.array([0.0, 1.0]), np.array([1.0]))  # length mismatch
