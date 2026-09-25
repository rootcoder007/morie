"""Tests for jolagf.joseph_lag_feature."""

import pytest

from morie.fn.jolagf import joseph_lag_feature


def test_jolagf_basic():
    """Rows start where every lag exists: y_t with (y_{t-1}, y_{t-3})."""
    r = joseph_lag_feature([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1])
    assert r["lags"] == [1, 3]
    assert r["rows"] == [[3.0, 1.0], [4.0, 2.0]]
    assert r["target"] == [4.0, 5.0]


def test_jolagf_edge():
    with pytest.raises(ValueError, match="positive"):
        joseph_lag_feature([1.0, 2.0, 3.0], [0])
    with pytest.raises(ValueError, match="too short"):
        joseph_lag_feature([1.0, 2.0], [2])


