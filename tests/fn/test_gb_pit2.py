"""Tests for gb_pit2 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_pit2 import gibbons_pit_rng


def test_gb_pit2_basic():
    from morie.fn import _stats_core as stats
    rng = np.random.default_rng(2)
    draws = gibbons_pit_rng(rng.random(1500), stats.expon.ppf)["X"]
    assert stats.kstest(draws, "expon").pvalue > 0.01


def test_gb_pit2_edge():
    with pytest.raises(ValueError):
        gibbons_pit_rng([1.5], float)
