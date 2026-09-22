"""Tests for cqtmpl.cim_qtl."""

from morie.fn import _array_core as np

from morie.fn.cqtmpl import cim_qtl


def test_cqtmpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    left = np.random.default_rng(43).integers(0, 3, 100).astype(float)
    right = np.random.default_rng(44).integers(0, 3, 100).astype(float)
    r_left = 0.01
    r_right = 0.01
    cofactors = (np.random.default_rng(42).normal(0, 1, 100).astype(float),)
    result = cim_qtl(y, left, right, r_left, r_right, cofactors)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "lod" in result
    assert "n" in result
    assert result["n"] == 100
    assert result["n_cofactors"] == 1
    assert result["method"].startswith("composite interval mapping")
    assert isinstance(result["loglik_history"], list)
    assert len(result["loglik_history"]) >= 1
    assert isinstance(result["posterior"], list)
    assert len(result["posterior"]) == 100


def test_cqtmpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    left = np.random.default_rng(43).integers(0, 3, 100).astype(float)
    right = np.random.default_rng(44).integers(0, 3, 100).astype(float)
    r_left = 0.01
    r_right = 0.01
    result = cim_qtl(y, left, right, r_left, r_right)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n_cofactors"] == 0
    assert result["n"] == 100
