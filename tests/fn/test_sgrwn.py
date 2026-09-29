"""Tests for morie.fn.sgrwn: row standardisation recomputed."""

from morie.fn.sgrwn import row_normalize_weights


def test_rows_sum_to_one_and_zero_rows_stay_zero():
    W = [[0, 2, 2], [1, 0, 3], [0, 0, 0]]
    r = row_normalize_weights(W)
    Wn = r.extra["W_normalized"].tolist()
    assert Wn[0] == [0.0, 0.5, 0.5]
    assert Wn[1] == [0.25, 0.0, 0.75]
    assert Wn[2] == [0.0, 0.0, 0.0]
    assert abs(r.statistic - 2 / 3) < 1e-15
