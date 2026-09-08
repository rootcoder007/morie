"""Tests for alnsmp.alammar_negative_sampling_skipgram."""

from morie.fn.alnsmp import alammar_negative_sampling_skipgram


def test_alnsmp_basic():
    out = alammar_negative_sampling_skipgram([100.0], [1.0], [[-1.0]])
    assert abs(out["estimate"]) < 1e-8


def test_alnsmp_edge():
    import pytest
    with pytest.raises(ValueError, match="dimension"):
        alammar_negative_sampling_skipgram([1.0], [1.0, 2.0], [[1.0]])
