"""Tests for gb734 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb734 import gibbons_linrank_symmetry_cond


def test_gb734_basic():
    out = gibbons_linrank_symmetry_cond(np.arange(1, 9))
    assert out["symmetric"] is True and out["constant"] == 9.0


def test_gb734_edge():
    assert gibbons_linrank_symmetry_cond([1.0, 5.0, 2.0])["symmetric"] is False
    with pytest.raises(ValueError):
        gibbons_linrank_symmetry_cond([1.0])
