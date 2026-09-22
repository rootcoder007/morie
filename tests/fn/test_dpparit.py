"""Tests for dpparit.pitman_yor_process."""

from morie.fn.dpparit import pitman_yor_process


def test_dpparit_basic():
    """Test basic functionality."""
    n = 100
    alpha = 0.05
    sigma = 0.5
    result = pitman_yor_process(n, alpha, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "K" in result
    assert "counts" in result
    assert "p_new" in result
    assert "n" in result
    # n is echoed back
    assert result["n"] == n
    # K equals number of occupied tables (length of counts) and the
    # documented estimate key holds that same value.
    assert result["K"] == len(result["counts"])
    assert result["estimate"] == result["K"]
    # p_new matches the documented Pitman-Yor formula evaluated on
    # the post-seating state (K = len(counts) at the end of the loop).
    expected_p_new = (alpha + sigma * len(result["counts"])) / (n + alpha)
    assert result["p_new"] == expected_p_new


def test_dpparit_edge():
    """Test edge cases."""
    n = 100
    alpha = 0.05
    sigma = 0.5
    result = pitman_yor_process(n, alpha, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result
