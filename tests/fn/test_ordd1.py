"""Tests for morie.fn.ordd1 — OTIS RDD age."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn._containers import ESRes
from morie.fn.ordd1 import otis_rdd_age


class TestOtisRddAge:
    def test_returns_esres(self):
        rng = np.random.default_rng(42)
        n = 200
        age = rng.uniform(14, 22, n)
        outcome = (age >= 18).astype(float) * 2.0 + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"outcome": outcome, "age": age})
        result = otis_rdd_age(df)
        assert isinstance(result, ESRes)

    def test_positive_jump(self):
        rng = np.random.default_rng(42)
        n = 300
        age = rng.uniform(15, 21, n)
        outcome = (age >= 18).astype(float) * 5.0 + rng.standard_normal(n) * 0.3
        df = pd.DataFrame({"outcome": outcome, "age": age})
        result = otis_rdd_age(df)
        assert result.estimate > 1.0
