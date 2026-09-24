"""Tests for bayes_estimate_binomial.bayes_estimate_binomial."""

from morie.fn import _array_core as np

from morie.fn.bayes_estimate_binomial import (
    bayes_estimate_binomial,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e24_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = int(rng.integers(10, 200))
    w = int(rng.integers(0, n + 1))
    a = float(rng.uniform(0.5, 5.0))
    b = float(rng.uniform(0.5, 5.0))
    result = bayes_estimate_binomial(w, n, a, b)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e24_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n = int(rng.integers(5, 50))
    w = int(rng.integers(0, n + 1))
    a = float(rng.uniform(0.1, 10.0))
    b = float(rng.uniform(0.1, 10.0))
    result = bayes_estimate_binomial(w, n, a, b)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0
