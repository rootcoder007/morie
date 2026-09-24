"""Tests for pheno2.phenotype_qc."""

import math

from morie.fn import _array_core as np

from morie.fn.pheno2 import phenotype_qc


def test_pheno2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y = rng.uniform(0.1, 10.0, 100)
    result = phenotype_qc(y)
    assert isinstance(result, dict)
    expected_keys = {
        "estimate", "loglik", "n_out", "flags", "lower",
        "upper", "transformed", "n", "method"
    }
    for key in expected_keys:
        assert key in result
    n = len(y)
    assert len(result["flags"]) == n
    assert len(result["transformed"]) == n
    assert isinstance(result["n_out"], int)
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["loglik"])
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
    assert result["n"] == n


def test_pheno2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    y = rng.uniform(0.5, 5.0, 40)
    result = phenotype_qc(y, k=2.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 40
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
