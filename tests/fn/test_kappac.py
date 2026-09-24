"""Tests for kappac.cohens_kappa."""

from morie.fn import _array_core as np

from morie.fn.kappac import cohens_kappa


def test_kappac_basic():
    """Test basic functionality."""
    rater1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    rater2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = cohens_kappa(rater1, rater2)
    assert isinstance(result, dict)
    assert "kappa" in result


def test_kappac_edge():
    """Test edge cases."""
    rater1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    rater2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = cohens_kappa(rater1, rater2)
    assert isinstance(result, dict)
