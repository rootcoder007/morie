"""Tests for gb251 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb251 import gibbons_pit


def test_gb251_basic():
    rng = np.random.default_rng(1)
    assert gibbons_pit(rng.standard_normal(300))["ks_p"] > 0.01


def test_gb251_edge():
    with pytest.raises(ValueError):
        gibbons_pit([np.inf])
