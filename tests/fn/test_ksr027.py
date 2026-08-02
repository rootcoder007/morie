"""Tests for ksr027 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr027 import kosorok_ch2_law_large_numbers_pointwise


def test_ksr027_basic():
    rng = np.random.default_rng(3)
    out = kosorok_ch2_law_large_numbers_pointwise(rng.random(2000), 0.4)
    assert out["shrinking"] is True


def test_ksr027_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_law_large_numbers_pointwise([0.1, 0.2], 0.4)
