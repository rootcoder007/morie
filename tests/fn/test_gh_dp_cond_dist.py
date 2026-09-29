"""Tests for gh_dp_cond_dist.ghosal_dp_conditional_distribution."""

from morie.fn import _array_core as np
from morie.fn.gh_dp_cond_dist import ghosal_dp_conditional_distribution


def test_gh_dp_cond_dist_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_conditional_distribution(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_dp_cond_dist_edge():
    """Test edge cases."""
    result = ghosal_dp_conditional_distribution(np.array([42.0]))
    assert np.isscalar(result["estimate"]) or np.asarray(result["estimate"]).shape == ()
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_conditional_dp_draw_is_a_normalised_gamma_vector():
    import pytest

    from morie.fn import _array_core as np

    a = [0.5, 1.5, 2.0]
    rng = np.random.default_rng(8)
    g = [float(rng.gamma(v, 1.0)) for v in a]
    r = ghosal_dp_conditional_distribution(a, w=0.9, seed=8)
    assert r["P_cond"] == pytest.approx([v / sum(g) for v in g], rel=1e-13)
    # Theorem 4.5: the conditional law does not depend on w
    assert ghosal_dp_conditional_distribution(a, w=0.1, seed=8)["P_cond"] == r["P_cond"]
