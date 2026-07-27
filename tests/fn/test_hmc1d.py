"""Tests for hmc1d.geron_causal_1d_conv."""

import pytest

from morie.fn.hmc1d import geron_causal_1d_conv


def test_hmc1d_basic():
    result = geron_causal_1d_conv([1.0, 2.0, 3.0, 4.0], [1.0, 0.5])
    assert result["y"] == pytest.approx([1.0, 2.5, 4.0, 5.5])
    assert result["kernel_size"] == 2


def test_hmc1d_edge():
    # no future leakage: changing the last input leaves earlier outputs alone
    a = geron_causal_1d_conv([1.0, 2.0, 3.0], [1.0, 1.0])["y"]
    b = geron_causal_1d_conv([1.0, 2.0, 99.0], [1.0, 1.0])["y"]
    assert a[:2] == pytest.approx(b[:2])
    with pytest.raises(ValueError):
        geron_causal_1d_conv([1.0], [1.0, 1.0])  # kernel longer than input
