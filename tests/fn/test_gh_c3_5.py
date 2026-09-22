"""Tests for gh_c3_5.ghosal_countable_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_5 import ghosal_countable_dp


def test_gh_c3_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_countable_dp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c3_5_edge():
    """Test edge cases."""
    result = ghosal_countable_dp(np.array([42.0]))
    # The function does not return a key 'n'; it returns the documented payload
    # keys from GvdV 2017 eq. 3.4: estimate, p_cells, p_tail, alpha, alpha_tail.
    assert "estimate" in result
    assert "p_cells" in result
    assert "p_tail" in result
    assert "alpha" in result
    assert "alpha_tail" in result

    # With default alpha_total=5.0 and k=6, the alpha vector is
    # alpha_j = alpha_total / 2**(j+1) for j = 0..k-1, and
    # alpha_tail = alpha_total - sum(alpha_j).
    k = 6
    alpha_total = 5.0
    expected_alpha = [alpha_total / (2.0 ** (j + 1)) for j in range(k)]
    expected_alpha_tail = alpha_total - sum(expected_alpha)

    assert list(result["alpha"]) == expected_alpha
    assert result["alpha_tail"] == expected_alpha_tail

    # p_cells and p_tail are a length-k and length-1 partition of the unit
    # interval (Dirichlet normalisation of gammas, including the tail cell).
    p_cells = np.asarray(result["p_cells"], dtype=float)
    p_tail = float(result["p_tail"])
    assert p_cells.shape == (k,)
    assert np.all(np.isfinite(p_cells))
    assert np.isfinite(p_tail)
    assert np.all(p_cells >= 0.0)
    assert p_tail >= 0.0
    assert abs(float(np.sum(p_cells)) + p_tail - 1.0) < 1e-12
