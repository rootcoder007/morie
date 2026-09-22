"""Tests for edgrn.edger_diff."""

from morie.fn import _array_core as np

from morie.fn.edgrn import edger_diff


def test_edgrn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    lrt = 5.0
    q = 1
    quasi_dispersion = 1.0
    df_residual = 10.0
    result = edger_diff(lrt, q, quasi_dispersion, df_residual)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "F" in result
    assert "df1" in result
    assert "df2" in result
    assert "p_value" in result
    assert "lrt_p_value" in result
    # Independent arithmetic from the documented formula.
    expected_F = lrt / (q * quasi_dispersion)
    expected_x = q * expected_F / (q * expected_F + df_residual)
    assert abs(result["F"] - expected_F) < 1e-12
    assert abs(result["estimate"] - expected_F) < 1e-12
    assert result["df1"] == q
    assert result["df2"] == df_residual
    # expected_x should be in (0, 1) for positive inputs.
    assert 0.0 < expected_x < 1.0
    # p_value should be in (0, 1).
    assert 0.0 < result["p_value"] < 1.0
    # lrt_p_value should be in (0, 1) for positive LRT.
    assert 0.0 < result["lrt_p_value"] < 1.0


def test_edgrn_edge():
    """Test edge cases with df_prior shrunken dispersion."""
    rng = np.random.default_rng(42)
    lrt = 3.5
    q = 2
    quasi_dispersion = 0.5
    df_residual = 8.0
    df_prior = 4.0
    result = edger_diff(lrt, q, quasi_dispersion, df_residual, df_prior=df_prior)
    assert isinstance(result, dict)
    # df2 should be df_residual + df_prior when df_prior is supplied.
    expected_df2 = df_residual + df_prior
    expected_F = lrt / (q * quasi_dispersion)
    assert abs(result["F"] - expected_F) < 1e-12
    assert result["df1"] == q
    assert abs(result["df2"] - expected_df2) < 1e-12
    assert 0.0 < result["p_value"] < 1.0
