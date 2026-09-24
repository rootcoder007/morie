"""Tests for morie.fn.mi — mutual information estimation."""

from morie.fn import _array_core as np

from morie.fn.mi import mutual_info


class TestMutualInfo:
    def test_correlated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = x + rng.normal(0, 0.1, 500)
        res = mutual_info(x, y)
        assert res.extra["mi_nats"] > 0.1

    def test_independent(self):
        # Correlated sample
        rng = np.random.default_rng(42)
        x_corr = rng.normal(0, 1, 500)
        y_corr = x_corr + rng.normal(0, 0.1, 500)
        res_corr = mutual_info(x_corr, y_corr)
        # Independent sample
        rng2 = np.random.default_rng(42)
        x_indep = rng2.normal(0, 1, 500)
        y_indep = rng2.normal(0, 1, 500)
        res_indep = mutual_info(x_indep, y_indep)
        assert res_indep.extra["mi_nats"] < res_corr.extra["mi_nats"]

    def test_nbins(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = x * 2 + rng.normal(0, 0.5, 200)
        res10 = mutual_info(x, y, n_bins=10)
        res30 = mutual_info(x, y, n_bins=30)
        assert res10.extra["mi_nats"] > 0
        assert res30.extra["mi_nats"] > 0
