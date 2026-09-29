"""Tests for morie.fn.boores -- bootstrap resampling."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.boores import boores, bootstrap_resample


class TestBoores:
    def test_alias(self):
        assert boores is bootstrap_resample

    def test_mean_bootstrap(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        result = bootstrap_resample(x, n_boot=500, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 10.0) < 1.0
        assert result.extra["ci_lower"] < 10.0 < result.extra["ci_upper"]

    def test_median(self):
        x = np.arange(1, 21, dtype=float)
        result = bootstrap_resample(x, stat_fn="median", n_boot=200, seed=42)
        assert abs(result.value - 10.5) < 3.0

    def test_bootstrap_summary_recomputed(self):
        """Replay the documented generator: rng.choice with replacement."""
        import math

        import pytest

        x = np.array([2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5])
        rng = np.random.default_rng(7)
        reps = [float(np.median(rng.choice(x, size=8, replace=True))) for _ in range(50)]
        r = bootstrap_resample(x, stat_fn="median", n_boot=50, seed=7)
        m = sum(reps) / 50
        assert r.value == pytest.approx(m, rel=1e-14)
        assert r.extra["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in reps) / 49), rel=1e-12)
        assert r.extra["bias"] == pytest.approx(m - float(np.median(x)), rel=1e-12, abs=1e-14)
