"""Tests for rng037.rangayyan_ch3_discrete_convolution_causal_alt."""

import pytest

from morie.fn.bsasig import rangayyan_ch3_discrete_convolution_causal
from morie.fn.bsasig import rangayyan_ch3_discrete_convolution_causal_alt


def test_rng037_basic():
    x, h = [1.0, 2.0, 3.0], [0.5, 0.25]
    a = rangayyan_ch3_discrete_convolution_causal_alt(x, h)
    b = rangayyan_ch3_discrete_convolution_causal(x, h)
    assert a["y"] == pytest.approx(b["y"])  # commutativity


def test_rng037_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_discrete_convolution_causal_alt([1.0], [], n=0)
