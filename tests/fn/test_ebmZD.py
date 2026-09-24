"""Tests for ebmZD.zonal_ebm."""

from morie.fn import _array_core as np

from morie.fn.ebmZD import zonal_ebm


def test_ebmZD_basic():
    """Test basic functionality."""
    result = zonal_ebm(1.0, start=20.0)
    assert isinstance(result, dict)
    expected_keys = {"temperature", "global_mean", "ice_fraction",
                     "albedo", "latitude", "converged", "snowball"}
    assert expected_keys.issubset(result.keys())
    assert len(result["temperature"]) == 9
    assert len(result["latitude"]) == 9
    assert not result["snowball"]
    assert result["global_mean"] > 0


def test_ebmZD_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    S = list(rng.uniform(0.5, 1.5, 9))
    result = zonal_ebm(S, start=-40.0)
    assert isinstance(result, dict)
    assert result["snowball"]
    assert result["global_mean"] < 0
    assert len(result["temperature"]) == 9
