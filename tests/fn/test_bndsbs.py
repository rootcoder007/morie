"""Tests for bndsbs.bound_subset_inference."""

from morie.fn import _array_core as np

from morie.fn.bndsbs import bound_subset_inference


def test_bndsbs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    theta_full = rng.normal(0, 1, (100, 3))
    subset_idx = np.array([0, 2])
    result = bound_subset_inference(theta_full, subset_idx)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert "total_width" in result
    assert "max_width" in result
    assert "d_subset" in result
    assert "m" in result
    assert "d" in result
    assert result["d_subset"] == 2
    assert result["m"] == 100
    assert result["d"] == 3

    # Independent computation of the projection bounds.
    cols = [[r[k] for r in theta_full] for k in subset_idx]
    lo0 = min(cols[0])
    hi0 = max(cols[0])
    assert result["lower"] == lo0
    assert result["upper"] == hi0
    assert result["width"] == hi0 - lo0
    widths = [max(c) - min(c) for c in cols]
    assert result["total_width"] == sum(widths)
    assert result["max_width"] == max(widths)


def test_bndsbs_edge():
    """Test edge cases."""
    theta_full = [
        [1.0, 2.0, 3.0],
        [4.0, 0.0, 6.0],
        [-1.0, 5.0, 2.0],
    ]
    subset_idx = np.array([1])
    result = bound_subset_inference(theta_full, subset_idx)
    assert isinstance(result, dict)
    # Only coordinate 1: min=0, max=5 -> width=5.
    assert result["lower"] == 0.0
    assert result["upper"] == 5.0
    assert result["width"] == 5.0
    assert result["total_width"] == 5.0
    assert result["max_width"] == 5.0
    assert result["d_subset"] == 1
    assert result["m"] == 3
    assert result["d"] == 3
