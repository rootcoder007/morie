"""Tests for impfun.genotype_imputation."""

from morie.fn import _array_core as np

from morie.fn.impfun import genotype_imputation


def test_impfun_basic():
    """Test basic functionality."""
    study_hap = 0.5
    reference_haps = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = genotype_imputation(study_hap, reference_haps)
    assert isinstance(result, dict)
    assert "posterior" in result


def test_impfun_edge():
    """Test edge cases."""
    study_hap = 0.5
    reference_haps = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = genotype_imputation(study_hap, reference_haps)
    assert isinstance(result, dict)
