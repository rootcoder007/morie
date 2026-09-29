"""Tests for jomimi.joseph_missing_data_imputation_ts."""

from morie.fn import _array_core as np
from morie.fn.jomimi import joseph_missing_data_imputation_ts


def test_jomimi_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_missing_data_imputation_ts(x)
    assert isinstance(result, dict)
    assert "x" in result


def test_jomimi_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_missing_data_imputation_ts(x)
    assert isinstance(result, dict)


def test_gap_filling_rules_recomputed():
    x = [1.0, None, None, 4.0, 5.0, None, 9.0, None]
    lin = joseph_missing_data_imputation_ts(x, method="linear")["x"]
    assert lin == [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 9.0]
    assert joseph_missing_data_imputation_ts(x, method="ffill")["x"][1:3] == [1.0, 1.0]
    gm = (1.0 + 4.0 + 5.0 + 9.0) / 4
    assert joseph_missing_data_imputation_ts(x, method="mean")["x"][7] == gm
    s = joseph_missing_data_imputation_ts(x, method="seasonal", season=2)["x"]
    # position 1 shares parity only with index 3; position 2 with 0, 4 and 6
    assert s[1] == 4.0 and s[2] == (1.0 + 5.0 + 9.0) / 3
