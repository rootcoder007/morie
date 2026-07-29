"""Tests for alspl.alammar_sampling_decoding."""

from morie.fn.alspl import alammar_sampling_decoding


def test_alspl_basic():
    out = alammar_sampling_decoding([[9.0, 0.0]], seed=1)
    assert out["tokens"] == [0]


def test_alspl_edge():
    a = alammar_sampling_decoding([[1.0, 1.0]] * 5, seed=3)
    b = alammar_sampling_decoding([[1.0, 1.0]] * 5, seed=3)
    assert a["tokens"] == b["tokens"]
