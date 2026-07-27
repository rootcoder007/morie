"""Tests for grc1d.geron_causal_1d_cnn."""

import numpy as np
import pytest

from morie.fn.grc1d import geron_causal_1d_cnn


def test_grc1d_basic():
    x = np.arange(1.0, 7.0)
    dil = geron_causal_1d_cnn(x, [1.0, 1.0], dilation=2)  # y_t = x_t + x_{t-2}
    assert dil["y"][2:] == pytest.approx(x[2:] + x[:-2])
    assert dil["receptive_field"] == 3


def test_grc1d_edge():
    x = np.arange(1.0, 7.0)
    strict = geron_causal_1d_cnn(x, [1.0], strict=True)  # y_t = x_{t-1}
    assert strict["y"][1:] == pytest.approx(x[:-1])
    with pytest.raises(ValueError):
        geron_causal_1d_cnn(x, [1.0, 1.0], dilation=0)
