"""Tests for ricei."""

import numpy as np
import pytest

from morie.fn.ricei import rice_index


def test_ricei_basic():
    V = np.array([[1, 1], [1, 1], [1, 1], [0, 1]], dtype=float)
    out = rice_index(V, np.array(["a", "a", "a", "a"]))
    assert out["matrix"]["a"] == pytest.approx([0.5, 1.0])


def test_ricei_edge():
    with pytest.raises(ValueError):
        rice_index(np.ones((4, 2)), ["a", "a"])  # length mismatch
    with pytest.raises(ValueError):
        rice_index(np.ones(4), ["a"] * 4)  # 1-D votes
