"""Tests for gh_ap_d2.ghosal_lecam_lemma."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_d2 import ghosal_lecam_lemma


def test_gh_ap_d2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    dtv = rng.uniform(0.0, 1.0)
    p0_phi = rng.uniform(0.0, 1.0 - dtv)
    prior_mass = rng.uniform(0.1, 2.0)
    integral = rng.uniform(0.0, 1.0)
    result = ghosal_lecam_lemma(dtv, p0_phi, prior_mass, integral)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "term_tv" in result
    assert "term_test" in result
    assert "term_prior" in result
    assert "informative" in result
    # Independent recomputation of the formula from the docstring.
    expected = dtv + p0_phi + integral / prior_mass
    assert abs(result["bound"] - expected) < 1e-12
    assert result["term_tv"] == dtv
    assert result["term_test"] == p0_phi
    assert abs(result["term_prior"] - integral / prior_mass) < 1e-12
    assert result["informative"] == (1.0 if expected < 1.0 else 0.0)


def test_gh_ap_d2_edge():
    """Test edge cases."""
    # All zero inputs: bound = 0, informative = 1 (0 < 1).
    result = ghosal_lecam_lemma(0.0, 0.0, 1.0, 0.0)
    assert isinstance(result, dict)
    assert result["bound"] == 0.0
    assert result["term_tv"] == 0.0
    assert result["term_test"] == 0.0
    assert result["term_prior"] == 0.0
    assert result["informative"] == 1.0
