"""Tests for icc3k.icc_two_way_mixed_avg."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.icc3k import icc_two_way_mixed_avg


def test_icc3k_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(123)
    n_subjects = 20
    k_raters = 3
    total = n_subjects * k_raters

    # Generate ratings
    y = rng.normal(0, 1, total)

    # Build crossed design: each subject seen by each rater exactly once
    subject = []
    rater = []
    for i in range(n_subjects):
        for j in range(k_raters):
            subject.append(i)
            rater.append(j)

    result = icc_two_way_mixed_avg(y, subject, rater)

    # The function returns a RichResult (dict-like)
    assert isinstance(result, dict)

    # Check required keys from the docstring
    required_keys = [
        "value", "icc_single", "icc2k", "max_rater_offset",
        "k", "n", "MSR", "MSC", "MSE", "case",
        "consistency_not_agreement", "method"
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # ICC value should be finite
    assert math.isfinite(result["value"])

    # k and n should match the design
    assert result["k"] == k_raters
    assert result["n"] == n_subjects

    # Case label
    assert result["case"] == "ICC(3,k)"

    # max_rater_offset should be finite
    assert math.isfinite(result["max_rater_offset"])


def test_icc3k_edge():
    """Test edge case: constant ratings lead to zero between-target MSR."""
    rng = np.random.default_rng(123)
    n_subjects = 10
    k_raters = 2
    total = n_subjects * k_raters

    # Constant ratings -> MSR = 0, should raise ValueError per docstring
    y = np.zeros(total)

    subject = []
    rater = []
    for i in range(n_subjects):
        for j in range(k_raters):
            subject.append(i)
            rater.append(j)

    with pytest.raises(ValueError):
        icc_two_way_mixed_avg(y, subject, rater)
