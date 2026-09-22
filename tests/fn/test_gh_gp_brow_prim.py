"""Tests for gh_gp_brow_prim.ghosal_gp_brownian_primitive."""

from morie.fn import _array_core as np

from morie.fn.gh_gp_brow_prim import ghosal_gp_brownian_primitive


def test_gh_gp_brow_prim_basic():
    """Test basic functionality."""
    result = ghosal_gp_brownian_primitive()
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "roughness_bm" in result
    assert np.all(np.isfinite(np.asarray(result["roughness_bm"], dtype=float)))
    # Plain arithmetic: roughness_bm = sum(|w[i+1]-w[i]|) / (n-1); the integrated
    # path should be strictly smoother, i.e. estimate <= roughness_bm / 2.
    rb = float(np.asarray(result["roughness_bm"], dtype=float))
    est = float(np.asarray(result["estimate"], dtype=float))
    assert est <= rb / 2.0 + 1e-12


def test_gh_gp_brow_prim_edge():
    """Test edge cases: single-primitive (k=1) returns expected keys."""
    result = ghosal_gp_brownian_primitive(k=1)
    assert "estimate" in result
    assert "roughness_bm" in result
    assert "smoother" in result
    assert "method" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    rb = float(np.asarray(result["roughness_bm"], dtype=float))
    assert np.isfinite(est)
    assert np.isfinite(rb)
