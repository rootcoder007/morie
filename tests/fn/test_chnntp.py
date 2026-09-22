"""Tests for chnntp.channel_capacity."""

from morie.fn import _array_core as np

from morie.fn.chnntp import channel_capacity


def test_chnntp_basic():
    """Test basic functionality."""
    # Binary symmetric channel with crossover probability 0.1.
    # Row x is the output distribution given input x.
    p = 0.1
    channel = [[1.0 - p, p],
               [p, 1.0 - p]]
    result = channel_capacity(channel)

    # For a BSC the capacity is 1 - H2(p), where H2(p) = -p log2 p - (1-p) log2(1-p).
    H2 = -(p * _log2(p) + (1.0 - p) * _log2(1.0 - p))
    expected_capacity = 1.0 - H2

    assert isinstance(result, dict)
    assert "capacity_bits" in result
    assert "capacity_nats" in result
    assert "input_dist" in result
    assert "trace" in result
    assert "iterations" in result

    # Computed independently from the documented formula.
    assert abs(result["capacity_bits"] - expected_capacity) < 1e-6

    # Mutual information in nats = capacity_bits * ln(2).
    assert abs(result["capacity_nats"] - result["capacity_bits"] * _ln(2)) < 1e-6

    # Optimal input distribution for a BSC is uniform.
    for px in result["input_dist"]:
        assert abs(px - 0.5) < 1e-6

    # The trace must be monotonically non-decreasing (coordinate ascent).
    trace = result["trace"]
    for a, b in zip(trace, trace[1:]):
        assert b >= a - 1e-12

    assert result["iterations"] == 200


def test_chnntp_edge():
    """Test edge cases on a noiseless and a completely-noisy channel."""
    # Noiseless binary channel: each input maps deterministically to its own output.
    noiseless = [[1.0, 0.0],
                 [0.0, 1.0]]
    result = channel_capacity(noiseless)
    # Capacity of a noiseless channel is log2(|X|) = 1 bit here.
    assert abs(result["capacity_bits"] - 1.0) < 1e-6
    for px in result["input_dist"]:
        assert abs(px - 0.5) < 1e-6

    # Completely noisy channel: both inputs give the same output distribution,
    # so X and Y are independent and the capacity must be ~0.
    noisy = [[0.5, 0.5],
             [0.5, 0.5]]
    result = channel_capacity(noisy)
    assert abs(result["capacity_bits"]) < 1e-6
    assert abs(result["capacity_nats"]) < 1e-6


def _log2(x):
    import math
    return math.log(x, 2)


def _ln(x):
    import math
    return math.log(x)
