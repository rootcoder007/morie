"""Tests for gb_hg2.gibbons_hodges_lehmann_2."""

from morie.fn import _array_core as np

from morie.fn.gb_hg2 import gibbons_hodges_lehmann_2


def test_gb_hg2_basic():
    """Test basic functionality with the documented example."""
    x = [1, 6, 7]
    y = [2, 4, 9, 10, 12]
    result = gibbons_hodges_lehmann_2(x, y)
    assert isinstance(result, dict)

    # All documented keys are present.
    for key in (
        "estimate", "ci", "coverage", "n_differences",
        "median_difference", "shift_plausible",
    ):
        assert key in result

    # Median of the 15 differences Y_j - X_i should be 3.0 by definition.
    diffs = sorted(b - a for a in x for b in y)
    expected_estimate = float(np.median(diffs))
    assert result["estimate"] == expected_estimate

    # n_differences is m * n.
    assert result["n_differences"] == len(x) * len(y)

    # median_difference is median(y) - median(x) (not the HL estimator).
    expected_md = float(np.median(y) - np.median(x))
    assert result["median_difference"] == expected_md

    # The sample spreads are equal in this constructed example, so the
    # shift assumption is plausible.
    assert result["shift_plausible"] is True


def test_gb_hg2_edge():
    """Test edge cases: empty input raises."""
    import pytest
    with pytest.raises(ValueError):
        gibbons_hodges_lehmann_2([], [1, 2, 3])
    with pytest.raises(ValueError):
        gibbons_hodges_lehmann_2([1, 2, 3], [])
