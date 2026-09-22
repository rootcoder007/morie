"""Tests for areN.asymptotic_relative_efficiency."""

from morie.fn import _array_core as np

from morie.fn.areN import asymptotic_relative_efficiency


def test_areN_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    var1 = 1.0 + rng.random()  # strictly positive variance
    var2 = 1.0 + rng.random()  # strictly positive variance
    n1 = 10.0 + rng.random() * 100.0
    n2 = 10.0 + rng.random() * 100.0

    result = asymptotic_relative_efficiency(var1, var2, n1=n1, n2=n2)

    # Documented return keys
    assert "are" in result
    assert "logare" in result
    assert "var1" in result
    assert "var2" in result
    assert "normalmedian" in result
    assert "normalhl" in result

    # Values are echoed back as provided
    assert result["var1"] == float(var1)
    assert result["var2"] == float(var2)

    # Independent computation of the documented formula
    expected_are = (var1 / n1) / (var2 / n2)
    expected_logare = np.log(expected_are)
    assert result["are"] == expected_are
    assert result["logare"] == expected_logare

    # Classical Gaussian benchmarks
    assert result["normalmedian"] == 2.0 / np.pi
    assert result["normalhl"] == 3.0 / np.pi


def test_areN_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)

    # Equal variances and sample sizes => ARE of 1, log of 0
    var = 1.0 + rng.random()
    n = 50.0 + rng.random() * 100.0
    result = asymptotic_relative_efficiency(var, var, n1=n, n2=n)
    assert result["are"] == 1.0
    assert result["logare"] == 0.0

    # Reversing var1/var2 inverts the ratio
    v1 = 2.0
    v2 = 1.0
    n = 100.0
    result12 = asymptotic_relative_efficiency(v1, v2, n1=n, n2=n)
    result21 = asymptotic_relative_efficiency(v2, v1, n1=n, n2=n)
    assert result12["are"] * result21["are"] == 1.0

    # Sample sizes that scale the efficiency exactly
    v1 = 1.0
    v2 = 1.0
    result = asymptotic_relative_efficiency(v1, v2, n1=4.0, n2=2.0)
    expected_are = (v1 / 4.0) / (v2 / 2.0)
    assert result["are"] == expected_are
