"""Tests for almteb.alammar_mteb_benchmark_score."""

from morie.fn.almteb import alammar_mteb_benchmark_score


def test_almteb_basic():
    out = alammar_mteb_benchmark_score({"a": 1.0, "b": 0.0},
                                       {"a": "x", "b": "y"})
    assert out["estimate"] == 0.5


def test_almteb_edge():
    import pytest
    with pytest.raises(ValueError, match="no category"):
        alammar_mteb_benchmark_score({"a": 1.0}, {})
