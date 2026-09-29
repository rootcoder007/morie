"""Tests for morie.fn.swspars."""

from morie.fn.swspars import swspars

W = [[0, 0.6, 0.05, 0.35], [0.6, 0, 0.3, 0.1], [0.05, 0.3, 0, 0.65], [0.35, 0.1, 0.65, 0]]


def test_threshold():
    S = swspars(W, thr=0.2)
    assert [[v if v >= 0.2 else 0.0 for v in r] for r in W] == S
    R = swspars(W, thr=0.2, row_standardize=True)
    assert all(abs(sum(r) - 1.0) < 1e-15 for r in R)
    assert abs(R[0][1] - 0.6 / 0.95) < 1e-15
