"""Tests for morie.fn.mmaxn: values recomputed from the definition."""

from morie.fn.mmaxn import minmax_normalization


def test_unit_interval_map():
    x = [2.0, 4.0, 3.0, 10.0]
    assert minmax_normalization(x)["x_norm"] == [(v - 2.0) / 8.0 for v in x]


def test_columns_and_constant_column():
    X = [[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]]
    r = minmax_normalization(X)
    assert r["x_norm"] == [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]
    assert r["min"] == [1.0, 5.0]
