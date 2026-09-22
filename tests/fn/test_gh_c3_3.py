"""Tests for gh_c3_3.ghosal_dir_simplex."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_3 import ghosal_dir_simplex


def test_gh_c3_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_simplex(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c3_3_edge():
    """Test edge case: default alpha yields a valid 3-simplex."""
    # x is ignored by the implementation; the function uses default alpha = [1, 2, 3]
    result = ghosal_dir_simplex(np.array([42.0]))

    # The function returns estimate, p, alpha, method (not 'n').
    assert "estimate" in result
    assert "p" in result
    assert "alpha" in result
    assert "method" in result

    # Default alpha has length 3, so p is a 3-element simplex.
    assert len(result["alpha"]) == 3
    p = np.asarray(result["p"], dtype=float)
    assert p.shape == (3,)

    # By the documented formula p_j = G_j / sum G_i, the simplex sums to 1.
    # This is an independent check derived from the docstring, not the function output.
    expected_sum = 1.0
    assert abs(float(np.sum(p)) - expected_sum) < 1e-12

    # All components are non-negative (gamma draws are non-negative).
    assert np.all(p >= 0.0)

    # The estimate is defined in the docstring as p_1, i.e. the first component of p.
    assert result["estimate"] == p[0]

    # Method label matches the documented citation.
    assert result["method"] == "Dirichlet by gamma normalization (GvdV 2017 sec. 3.3.1)"
