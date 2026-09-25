"""Tests for si_proportion.si_proportion (Brus 2022, eq. 3.6)."""

import pytest

from morie.fn.si_proportion import si_proportion


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e6_basic():
    """p_hat = (1/n) sum y_k over 0/1 indicators."""
    y = [1, 0, 1, 1, 0, 0, 1, 1]
    result = si_proportion(y)
    assert isinstance(result, dict)
    assert result["value"] == 5 / 8


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e6_edge():
    """All zeros give 0; anything other than 0/1 is refused."""
    assert si_proportion([0, 0, 0])["value"] == 0.0
    with pytest.raises(ValueError):
        si_proportion([0.3, 1.0])
