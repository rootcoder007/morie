"""Tests for morie.fn.mblbt -- Moving block bootstrap."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mblbt import moving_block_bootstrap


class TestMovingBlockBootstrap:
    def test_basic_mean(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        r = moving_block_bootstrap(x)
        assert abs(r["estimate"] - np.mean(x)) < 1e-10
        assert r["se"] > 0
        assert r["ci_lower"] < r["ci_upper"]

    def test_explicit_block_size(self):
        x = np.arange(20, dtype=float)
        r = moving_block_bootstrap(x, block_size=5, n_boot=100)
        assert r["block_size"] == 5

    def test_median_statistic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        r = moving_block_bootstrap(x, statistic="median")
        assert r["statistic"] == "median"

    def test_boot_distribution_length(self):
        x = np.arange(30, dtype=float)
        r = moving_block_bootstrap(x, n_boot=200)
        assert len(r["boot_distribution"]) == 200

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least 4"):
            moving_block_bootstrap(np.array([1.0, 2.0, 3.0]))

    def test_invalid_statistic(self):
        with pytest.raises(ValueError, match="statistic must be"):
            moving_block_bootstrap(np.arange(10, dtype=float), statistic="bad")


def test_moving_blocks_replayed():
    import math

    x = np.array([0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1, 0.8, 0.6])
    n, L, B = 10, 3, 20
    xs = [float(v) for v in x]
    rng = np.random.default_rng(12)
    vals = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, size=math.ceil(n / L))
        samp = [v for s in starts for v in xs[int(s) : int(s) + L]][:n]
        vals.append(sum(samp) / n)
    m = sum(vals) / B
    r = moving_block_bootstrap(x, block_size=L, n_boot=B, seed=12)
    assert r["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in vals) / (B - 1)), rel=1e-12)
