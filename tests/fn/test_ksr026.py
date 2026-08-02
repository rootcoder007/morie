"""Tests for ksr026 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr026 import kosorok_ch2_empirical_distribution_function


def test_ksr026_basic():
    out = kosorok_ch2_empirical_distribution_function([0.1, 0.4, 0.7])
    assert out["F_n"][-1] == pytest.approx(1.0)


def test_ksr026_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_empirical_distribution_function([])
