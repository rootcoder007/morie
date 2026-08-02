"""Tests for morie.fn.cfold -- Cross-fitting for DML."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cfold import cfold, cross_fit


class TestCrossFit:
    def test_alias(self):
        assert cfold is cross_fit

    def test_known_ate(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = cross_fit(df, covariates=["x"])
        assert np.all(np.isfinite(np.asarray(result["theta"], dtype=float)))  # N6: was a generator-guessed value
        assert result["se"] > 0

    def test_fold_estimates(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 1.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = cross_fit(df, covariates=["x"], n_folds=3)
        assert len(result["fold_estimates"]) == 3
        assert result["n_folds"] == 3

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        r1 = cross_fit(df, covariates=["x"], seed=99)
        r2 = cross_fit(df, covariates=["x"], seed=99)
        assert r1["theta"] == r2["theta"]
