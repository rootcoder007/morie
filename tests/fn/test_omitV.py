"""Tests for omitV.omitted_variable_bias."""

from morie.fn.omitV import omitted_variable_bias


def test_omitV_basic():
    """Test basic functionality."""
    result = omitted_variable_bias()
    assert isinstance(result, dict)
    assert "bias" in result


def test_omitV_edge():
    """Test edge cases."""
    result = omitted_variable_bias()
    assert isinstance(result, dict)


def test_partial_r2_bias_and_adjusted_se_recomputed():
    """Cinelli-Hazlett (2020) eqs (12)-(14)."""
    import math

    import pytest

    est, se, df, ry, rd = 2.0, 0.5, 100, 0.1, 0.2
    bias = se * math.sqrt(ry * rd / (1 - rd)) * math.sqrt(df)
    adj_se = se * math.sqrt((1 - ry) / (1 - rd) * df / (df - 1))
    bf = math.sqrt(ry) * math.sqrt(rd / (1 - rd))
    r = omitted_variable_bias(estimate=est, se=se, df=df, r2_yz=ry, r2_dz=rd)
    assert r["bias"] == pytest.approx(bias, rel=1e-13)
    assert r["adjusted_se"] == pytest.approx(adj_se, rel=1e-13)
    assert r["adjusted_estimate"] == pytest.approx(est - bias, rel=1e-13)
    assert r["relative_bias"] == pytest.approx(bf / (est / se / math.sqrt(df)), rel=1e-13)
    assert omitted_variable_bias(delta=0.3, gamma=-2.0)["bias"] == pytest.approx(-0.6, rel=1e-15)
