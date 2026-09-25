"""Tests for weighted_category_total.weighted_category_total."""

import pytest

from morie.fn.weighted_category_total import (
    weighted_category_total,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e7_basic():
    """Survey-weighted count is the sum of weights over matching units."""
    weights = [2.5, 4.0, 1.5, 3.0, 2.0, 5.5]
    ys = ["a", "b", "a", "c", "b", "a"]

    for cat in ("a", "b", "c"):
        expected = sum(w for w, y in zip(weights, ys) if y == cat)
        result = weighted_category_total(weights, ys, cat)
        assert result["value"] == pytest.approx(expected, rel=1e-12)

    # eq. (6.7) partitions the sample: the category totals add to sum(w).
    total = sum(weighted_category_total(weights, ys, c)["value"]
                for c in ("a", "b", "c"))
    assert total == pytest.approx(sum(weights), rel=1e-12)

    # 2.5 + 1.5 + 5.5 = 9.5 for category "a", by arithmetic.
    assert weighted_category_total(weights, ys, "a")["value"] == pytest.approx(
        9.5, rel=1e-12)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e7_edge():
    """Absent category gives zero; unit weights reduce to a plain count."""
    weights = [1.0, 1.0, 1.0, 1.0]
    ys = [0, 1, 1, 1]

    assert weighted_category_total(weights, ys, 2)["value"] == 0.0
    assert weighted_category_total(weights, ys, 1)["value"] == pytest.approx(
        float(ys.count(1)), rel=1e-12)
    assert weighted_category_total(weights, ys, 0)["value"] == pytest.approx(
        1.0, rel=1e-12)

    # A zero weight contributes nothing even when the label matches.
    assert weighted_category_total([0.0, 3.0], [5, 5], 5)["value"] == (
        pytest.approx(3.0, rel=1e-12))

    with pytest.raises(ValueError):
        weighted_category_total([1.0, -1.0], [0, 1], 0)
    with pytest.raises(ValueError):
        weighted_category_total([1.0, 2.0], [0], 0)
