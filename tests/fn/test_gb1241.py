"""Tests for gb1241 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1241 import gibbons_concordance_w


def test_gb1241_basic():
    R = np.tile(np.arange(1, 6), (3, 1))
    out = gibbons_concordance_w(R)
    assert out["W"] == pytest.approx(1.0)
    assert out["mean_spearman"] == pytest.approx(1.0)


def test_gb1241_edge():
    with pytest.raises(ValueError):
        gibbons_concordance_w(np.arange(4.0))
