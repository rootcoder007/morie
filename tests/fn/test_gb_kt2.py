"""Tests for gb_kt2 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_kt2 import gibbons_kendall_exact


def test_gb_kt2_basic():
    out = gibbons_kendall_exact(5)
    assert out["mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["var"] == pytest.approx(2 * 15 / (9 * 20), abs=1e-12)  # 2(2n+5)/(9n(n-1))


def test_gb_kt2_edge():
    with pytest.raises(ValueError):
        gibbons_kendall_exact(12)  # enumeration bound
