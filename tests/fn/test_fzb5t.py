"""Tests for fzb5t.fauzi_b5_coefficient_mrl."""

from morie.fn import _array_core as np

from morie.fn.fzb5t import fauzi_b5_coefficient_mrl


def test_fzb5t_basic():
    """Test basic functionality."""
    dg = 0.5
    density = 0.3
    mrl = 2.0
    surv = 0.8
    result = fauzi_b5_coefficient_mrl(dg, density, mrl, surv)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "varterm" in result
    assert "method" in result
    expected_estimate = dg * density * mrl ** 2
    expected_varterm = expected_estimate / (surv * surv)
    assert result["estimate"] == expected_estimate
    assert result["varterm"] == expected_varterm


def test_fzb5t_no_surv():
    """Test without surv argument returns NaN varterm."""
    dg = 0.5
    density = 0.3
    mrl = 2.0
    result = fauzi_b5_coefficient_mrl(dg, density, mrl)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "varterm" in result
    assert "method" in result
    expected_estimate = dg * density * mrl ** 2
    assert result["estimate"] == expected_estimate
    assert np.isnan(result["varterm"])


def test_fzb5t_edge():
    """Test edge cases."""
    dg = 1.0
    density = 0.0
    mrl = 2.0
    surv = 0.5
    result = fauzi_b5_coefficient_mrl(dg, density, mrl, surv)
    assert isinstance(result, dict)
    assert result["estimate"] == 0.0
