"""Tests for morie.fn.chrnf -- Chernoff distribution."""

import pytest

from morie.fn import _array_core as np
from morie.fn.chrnf import chernoff_distribution


class TestChernoffDistribution:
    def test_basic_output(self):
        x = np.linspace(-2, 2, 50)
        r = chernoff_distribution(x)
        assert len(r["cdf"]) == 50
        assert len(r["pdf"]) == 50

    def test_cdf_monotone(self):
        x = np.linspace(-3, 3, 100)
        r = chernoff_distribution(x)
        cdf = np.array(r["cdf"])
        assert np.all(np.diff(cdf) >= -1e-6)

    def test_cdf_bounds(self):
        x = np.linspace(-5, 5, 100)
        r = chernoff_distribution(x, n_grid=300)
        assert r["cdf"][0] < 0.1
        assert r["cdf"][-1] > 0.9

    def test_mean_near_zero(self):
        x = np.linspace(-3, 3, 50)
        r = chernoff_distribution(x, n_grid=200)
        assert abs(r["mean"]) < 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            chernoff_distribution(np.array([]))


def test_chernoff_monte_carlo_moments_match_groeneboom_wellner():
    """Chernoff's distribution has mean 0 and variance 0.26355 (Groeneboom
    and Wellner 2001, Table 1); the Monte Carlo with a fine grid recovers
    both to Monte Carlo accuracy (5000 argmaxes, sd of the mean ~0.007)."""
    import pytest

    r = chernoff_distribution([0.0], n_grid=400)
    assert r["mean"] == pytest.approx(0.0, abs=0.03)
    assert r["variance"] == pytest.approx(0.26355, abs=0.02)
    assert r["cdf"][0] == pytest.approx(0.5, abs=0.03)
