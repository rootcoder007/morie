"""Tests for fzb4t.fauzi_b4_coefficient_mrl."""

from morie.fn import _array_core as np

from morie.fn.fzb4t import fauzi_b4_coefficient_mrl


def test_fzb4t_basic():
    """Test basic functionality with scalar inputs."""
    surv = 0.5
    cumsurv = 1.2
    mrl = cumsurv / surv  # = 2.4
    result = fauzi_b4_coefficient_mrl(surv, cumsurv, mrl)
    # Per docstring, estimate = 2*cumsurv - surv*mrl^2
    expected_estimate = 2.0 * cumsurv - surv * mrl * mrl
    # varterm = estimate / surv^2
    expected_varterm = expected_estimate / (surv * surv)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["estimate"] == expected_estimate
    assert result["mrl"] == mrl
    assert result["varterm"] == expected_varterm
    assert "method" in result


def test_fzb4t_default_mrl():
    """Test that mrl defaults to cumsurv / surv."""
    surv = 0.4
    cumsurv = 0.8
    result = fauzi_b4_coefficient_mrl(surv, cumsurv)
    expected_mrl = cumsurv / surv  # = 2.0
    expected_estimate = 2.0 * cumsurv - surv * expected_mrl * expected_mrl
    expected_varterm = expected_estimate / (surv * surv)
    assert result["mrl"] == expected_mrl
    assert result["estimate"] == expected_estimate
    assert result["varterm"] == expected_varterm


def test_fzb4t_edge():
    """Test edge cases with scalar inputs."""
    surv = 1.0
    cumsurv = 0.0
    result = fauzi_b4_coefficient_mrl(surv, cumsurv)
    # mrl = 0 / 1 = 0; estimate = 2*0 - 1*0 = 0; varterm = 0/1 = 0
    assert result["estimate"] == 0.0
    assert result["mrl"] == 0.0
    assert result["varterm"] == 0.0
    assert isinstance(result, dict)
