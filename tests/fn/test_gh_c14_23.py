"""Tests for gh_c14_23.ghosal_ibp_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_23 import ghosal_ibp_def


def test_gh_c14_23_basic():
    """Test basic functionality."""
    n_customers = 30
    alpha = 3.0
    seed = 42
    result = ghosal_ibp_def(n_customers=n_customers, alpha=alpha, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independently compute expected_dishes using the documented formula alpha * H_n
    Hn = sum(1.0 / i for i in range(1, n_customers + 1))
    assert abs(result["expected_dishes"] - alpha * Hn) < 1e-12
    assert result["method"].startswith("Indian buffet process")


def test_gh_c14_23_edge():
    """Test edge cases."""
    n_customers = 1
    alpha = 3.0
    result = ghosal_ibp_def(n_customers=n_customers, alpha=alpha, seed=7)
    # Documented expected_dishes for n=1 is alpha * 1
    assert abs(result["expected_dishes"] - alpha * 1.0) < 1e-12
    assert int(result["estimate"]) >= 0
