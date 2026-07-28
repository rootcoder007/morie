"""Tests for gb_sp2 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_sp2 import gibbons_spearman_exact


def test_gb_sp2_basic():
    out = gibbons_spearman_exact(5)
    assert out["var"] == pytest.approx(0.25, abs=1e-12)  # exactly 1/(n-1)


def test_gb_sp2_edge():
    with pytest.raises(ValueError):
        gibbons_spearman_exact(12)
