"""Tests for gb921m.gibbons_mood_moments."""

from morie.fn.gb921m import gibbons_mood_moments


def test_gb921m_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    result = gibbons_mood_moments(m, n)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "var" in result
    assert "sd" in result
    assert "N" in result
    assert "m" in result
    assert "n" in result
    assert "method" in result

    N = m + n
    expected_mean = m * (N * N - 1) / 12.0
    expected_var = m * n * (N + 1) * (N * N - 4) / 180.0
    expected_sd = expected_var ** 0.5

    assert result["mean"] == expected_mean
    assert result["var"] == expected_var
    assert result["sd"] == expected_sd
    assert result["N"] == N
    assert result["m"] == m
    assert result["n"] == n


def test_gb921m_edge():
    """Test edge cases."""
    m = 1
    n = 1
    result = gibbons_mood_moments(m, n)
    assert isinstance(result, dict)

    N = m + n
    expected_mean = m * (N * N - 1) / 12.0
    expected_var = m * n * (N + 1) * (N * N - 4) / 180.0

    assert result["mean"] == expected_mean
    assert result["var"] == expected_var
    assert result["N"] == N
    assert result["m"] == m
    assert result["n"] == n
