"""Tests for rng036.rangayyan_ch3_discrete_convolution_causal."""

import pytest

from morie.fn.rng036 import rangayyan_ch3_discrete_convolution_causal


def test_rng036_basic():
    result = rangayyan_ch3_discrete_convolution_causal([1.0, 2.0, 3.0], [1.0, 1.0])
    assert result["y"] == pytest.approx([1.0, 3.0, 5.0, 3.0])
    assert rangayyan_ch3_discrete_convolution_causal([1.0, 2.0], [1.0], n=1)["value"] == 2.0


def test_rng036_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_discrete_convolution_causal([], [1.0])
    with pytest.raises(ValueError):
        rangayyan_ch3_discrete_convolution_causal([1.0, 2.0], [1.0], n=5)  # out of range
