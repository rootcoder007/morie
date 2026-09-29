"""Tests for pdic.effective_parameters_dic."""

from morie.fn import _array_core as np
from morie.fn.pdic import effective_parameters_dic


def test_pdic_basic():
    """Test basic functionality."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_dic(deviance)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_pdic_edge():
    """Test edge cases."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_dic(deviance)
    assert isinstance(result, dict)


def test_effective_parameters_recomputed():
    import pytest

    d = [102.0, 98.5, 101.2, 99.8, 100.4, 97.9]
    db = sum(d) / 6
    pv = 0.5 * sum((v - db) ** 2 for v in d) / 5
    assert effective_parameters_dic(d)["estimate"] == pytest.approx(pv, rel=1e-13)
    r = effective_parameters_dic(d, d_at_mean=95.5)
    assert r["p_d"] == pytest.approx(db - 95.5, rel=1e-13) and r["variant"] == "p_D"
