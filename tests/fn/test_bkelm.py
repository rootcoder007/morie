"""Tests for bkelm.burkov_elman_rnn."""

from morie.fn.bkelm import burkov_elman_rnn


def test_bkelm_basic():
    out = burkov_elman_rnn([1.0], [0.0], [[0.0]], [[0.0]], [[1.0]],
                           [0.0], [0.0])
    assert out["h"] == [0.0]
    assert out["y"] == [0.0]


def test_bkelm_edge():
    import pytest
    with pytest.raises(ValueError, match="Wh must be"):
        burkov_elman_rnn([1.0], [0.0, 0.0], [[0.0]], [[0.0]], [[1.0]],
                         [0.0], [0.0])
