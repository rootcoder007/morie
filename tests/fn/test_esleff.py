"""Tests for esleff.esl_effective_dof."""

from morie.fn import _array_core as np

from morie.fn.esleff import esl_effective_dof


def test_esleff_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    S = rng.normal(0, 1, (n, n))
    result = esl_effective_dof(S)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    assert "trace_ssT" in result
    assert "df_variance" in result
    assert "is_projection" in result
    assert result["n"] == n
    # Independent expectation from the documented formula.
    expected_estimate = float(np.trace(S))
    expected_trace_ssT = float(np.trace(S @ S.T))
    expected_df_variance = float(np.trace(2.0 * S - S @ S.T))
    assert np.allclose(result["estimate"], expected_estimate)
    assert np.allclose(result["trace_ssT"], expected_trace_ssT)
    assert np.allclose(result["df_variance"], expected_df_variance)
    expected_is_projection = bool(
        np.allclose(S, S.T) and np.allclose(S @ S, S)
    )
    assert result["is_projection"] == expected_is_projection


def test_esleff_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    S = rng.normal(0, 1, (n, n))
    result = esl_effective_dof(S)
    assert isinstance(result, dict)
    assert "estimate" in result
