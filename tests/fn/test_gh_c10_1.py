"""Tests for gh_c10_1.ghosal_adapt_thm."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_1 import ghosal_adapt_thm


def test_gh_c10_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_adapt_thm(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "model_posterior" in result
    assert "K_true" in result
    assert "method" in result
    assert result["K_true"] == 3
    assert result["method"] == "adaptive model prior (GvdV 2017 sec. 10.1)"
    # estimate should be a nonnegative integer (model index in 0..K_max)
    est = float(result["estimate"])
    assert 0.0 <= est <= 12.0
    assert est == int(est)
    post = np.asarray(result["model_posterior"], dtype=float)
    assert post.shape[0] == 13  # K_max + 1 entries
    s = float(np.sum(post))
    assert abs(s - 1.0) < 1e-12
    assert np.all(post >= 0.0)


def test_gh_c10_1_edge():
    """Test edge cases."""
    # Single observation: y must have shape compatible with the
    # documented behaviour; the function returns a result whose
    # estimate is a nonnegative integer index and whose model
    # posterior sums to 1.
    result = ghosal_adapt_thm(np.array([42.0]))
    assert "estimate" in result
    est = float(result["estimate"])
    assert 0.0 <= est <= 12.0
    assert est == int(est)
    post = np.asarray(result["model_posterior"], dtype=float)
    assert post.shape[0] == 13
    s = float(np.sum(post))
    assert abs(s - 1.0) < 1e-12
