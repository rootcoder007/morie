"""Tests for gh_ap_k2.ghosal_assouad_lemma."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_k2 import ghosal_assouad_lemma


def test_gh_ap_k2_basic():
    """Test basic functionality with the documented signature."""
    result = ghosal_assouad_lemma(m=8, per_coord_sep=0.1, affinity=0.8)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independent computation of the documented formula:
    # R_n >= (m/2) * sep * min affinity
    expected = 0.5 * 8 * 0.1 * 0.8
    assert np.isclose(float(result["estimate"]), expected)
    assert result["grows_with_m"] is True
    assert result["method"] == "Assouad lower bound (GvdV 2017 App K)"


def test_gh_ap_k2_custom_params():
    """Test with a different (m, per_coord_sep, affinity) triple."""
    m = 10
    sep = 0.2
    aff = 0.5
    result = ghosal_assouad_lemma(m=m, per_coord_sep=sep, affinity=aff)
    expected = 0.5 * m * sep * aff
    assert "estimate" in result
    assert np.isclose(float(result["estimate"]), expected)
    assert result["grows_with_m"] is True


def test_gh_ap_k2_edge():
    """Test edge case: minimal valid inputs (m=1)."""
    result = ghosal_assouad_lemma(m=1, per_coord_sep=0.1, affinity=0.8)
    # The function returns a RichResult with key 'estimate', not 'n'.
    assert "estimate" in result
    expected = 0.5 * 1 * 0.1 * 0.8
    assert np.isclose(float(result["estimate"]), expected)
    assert result["grows_with_m"] is True
