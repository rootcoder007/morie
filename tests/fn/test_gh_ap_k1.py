"""Tests for gh_ap_k1.ghosal_fano_ineq."""

from morie.fn import _array_core as np
from morie.fn.gh_ap_k1 import ghosal_fano_ineq


def test_gh_ap_k1_basic():
    """Test basic functionality."""
    M = 10
    mutual_info = 1.5
    result = ghosal_fano_ineq(M, mutual_info)
    assert isinstance(result, dict)
    # Compute expected values independently from the documented formula.
    import math
    log_M = math.log(M)
    log2 = math.log(2.0)
    raw = 1.0 - (mutual_info + log2) / log_M
    expected_bound = min(1.0, max(0.0, raw))
    expected_informative = 1.0 if raw > 0.0 else 0.0

    assert "bound" in result
    assert "raw_bound" in result
    assert "log_M" in result
    assert "informative" in result
    assert "M" in result
    assert result["M"] == float(M)
    assert result["log_M"] == log_M
    assert result["raw_bound"] == raw
    assert result["bound"] == expected_bound
    assert result["informative"] == expected_informative


def test_gh_ap_k1_edge():
    """Test edge cases."""
    # M = 2 with mutual_info = 0 should yield raw_bound of 0 and informative = 0.
    M = 2
    mutual_info = 0.0
    result = ghosal_fano_ineq(M, mutual_info)
    assert isinstance(result, dict)
    import math
    raw = 1.0 - (mutual_info + math.log(2.0)) / math.log(M)
    expected_bound = min(1.0, max(0.0, raw))
    expected_informative = 1.0 if raw > 0.0 else 0.0
    assert result["raw_bound"] == raw
    assert result["bound"] == expected_bound
    assert result["informative"] == expected_informative
    assert result["M"] == float(M)
