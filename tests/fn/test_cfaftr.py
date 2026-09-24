"""Tests for cfaftr.cfa_one_factor."""

import math

from morie.fn import _array_core as np

from morie.fn.cfaftr import cfa_one_factor


def test_cfaftr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    result = cfa_one_factor(X)
    assert isinstance(result, dict)
    assert result["p"] == 3
    assert len(result["loadings"]) == 3
    assert len(result["uniquenesses"]) == 3
    assert len(result["communality"]) == 3
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["fml"])
    assert math.isfinite(result["spearman"])
    assert result["n_iter"] >= 0


def test_cfaftr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 5))
    factor_structure = [1, 1, 1, 0, 0]
    result = cfa_one_factor(X, factor_structure)
    assert isinstance(result, dict)
    assert result["p"] == 5
    assert len(result["loadings"]) == 5
    assert len(result["uniquenesses"]) == 5
    assert len(result["communality"]) == 5
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
