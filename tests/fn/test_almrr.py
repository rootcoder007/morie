"""Tests for almrr.alammar_mean_reciprocal_rank."""

from morie.fn.almrr import alammar_mean_reciprocal_rank


def test_almrr_basic():
    assert alammar_mean_reciprocal_rank([[3, 1, 2]], [[1]])["estimate"] == 0.5


def test_almrr_edge():
    out = alammar_mean_reciprocal_rank([[9]], [[1]])
    assert out["queries_missed"] == 1
