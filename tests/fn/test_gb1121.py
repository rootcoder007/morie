"""Tests for gb1121 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1121 import gibbons_kendall_tau


def test_gb1121_basic():
    from morie.fn import _stats_core as stats
    rng = np.random.default_rng(0)
    x = rng.standard_normal(25); y = 0.5 * x + rng.standard_normal(25)
    assert gibbons_kendall_tau(x, y)["tau"] == pytest.approx(
        stats.kendalltau(x, y).statistic, abs=1e-12)


def test_gb1121_edge():
    assert gibbons_kendall_tau([1, 2, 3], [3, 2, 1])["tau"] == -1.0
    with pytest.raises(ValueError):
        gibbons_kendall_tau([1.0], [2.0])
