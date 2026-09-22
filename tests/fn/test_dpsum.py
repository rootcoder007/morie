"""Tests for dpsum.dp_sum."""

from morie.fn import _array_core as np

from morie.fn.dpsum import dp_sum


def test_dpsum_basic():
    """Test basic functionality with fixed seed and scalar bounds."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1.0
    result = dp_sum(x, 0, 10, epsilon, seed=0)
    # Result is a RichResult (dict-like) with documented keys
    assert isinstance(result, dict)
    assert "release" in result
    assert "noise_scale" in result
    assert result["noise_scale"] == (10 - 0) / 1.0
    assert result["sensitivity"] == 10.0
    assert result["epsilon"] == 1.0
    assert result["method"] == "dp_sum"


def test_dpsum_edge():
    """Test edge cases: scalar inputs produce correct clipped sum."""
    # Single value, in-bounds
    result = dp_sum([5.0], 0, 10, epsilon=1.0, seed=0)
    assert isinstance(result, dict)
    # Independent computation of the true sum (all in-bounds, no clipping)
    x = [5.0]
    assert result["true_sum"] == sum(min(max(v, 0), 10) for v in x)
    assert result["clipped_fraction"] == 0.0
    # noise_scale must be (b - a) / epsilon
    assert result["noise_scale"] == (10 - 0) / 1.0
    assert result["sensitivity"] == 10.0

    # Out-of-bounds value gets clipped
    result2 = dp_sum([1.0, 2.0, 50.0], 0, 10, epsilon=1.0, seed=0)
    clipped_x = [min(max(v, 0), 10) for v in [1.0, 2.0, 50.0]]
    assert result2["true_sum"] == sum(clipped_x)
    assert result2["clipped_fraction"] == 1 / 3
