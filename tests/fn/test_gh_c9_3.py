"""Tests for gh_c9_3.ghosal_bpoly_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_3 import ghosal_bpoly_crt


def test_gh_c9_3_basic():
    """Test basic functionality."""
    result = ghosal_bpoly_crt()
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    assert float(est) >= 0.0


def test_gh_c9_3_keys_and_shape():
    """Test return keys and that l1_by_n has one entry per n."""
    result = ghosal_bpoly_crt(ns=(100, 800, 6400), seed=42)
    expected_keys = {"estimate", "l1_by_n", "improving", "method"}
    assert expected_keys.issubset(set(result.keys()))
    l1 = np.asarray(result["l1_by_n"], dtype=float)
    assert l1.shape == (3,)
    assert np.all(np.isfinite(l1))
    assert np.all(l1 >= 0.0)
    assert float(l1[-1]) == float(result["estimate"])


def test_gh_c9_3_improves_with_n():
    """The L1 error to the triangular truth 2x should decrease as n grows."""
    result = ghosal_bpoly_crt(ns=(100, 800, 6400), seed=42)
    l1 = [float(v) for v in result["l1_by_n"]]
    assert l1[2] < l1[0]


def test_gh_c9_3_custom_ns():
    """Test that a custom ns sequence produces the expected number of error terms."""
    result = ghosal_bpoly_crt(ns=(200, 1600), seed=7)
    assert len(result["l1_by_n"]) == 2
    assert result["improving"] == (result["l1_by_n"][-1] < result["l1_by_n"][0])


def test_gh_c9_3_edge():
    """Test edge case: single-element ns still works and yields one error."""
    result = ghosal_bpoly_crt(ns=(100,), seed=1)
    assert len(result["l1_by_n"]) == 1
    assert float(result["estimate"]) >= 0.0
