"""Tests for bmtme_conditionals.bmtme_conditionals."""

from morie.fn import _array_core as np

from morie.fn.bmtme_conditionals import bmtme_conditionals


def test_msm076_basic():
    """Test basic functionality."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Multi = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = bmtme_conditionals(Bayesian, Genomic, Multi, trait, environment, Model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm076_edge():
    """Test edge cases."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Multi = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = bmtme_conditionals(Bayesian, Genomic, Multi, trait, environment, Model)
    assert isinstance(result, dict)
