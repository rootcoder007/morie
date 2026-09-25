"""Tests for jonhts.joseph_nhits."""

import pytest

from morie.fn.jonhts import joseph_nhits


def test_jonhts_basic():
    """y = (1, 2, 3, 4), H = 2, kernel 2: max-pooling gives (2, 4); ratio 0.5
    asks for one knot, so the forecast is the constant wf . (2, 4) = 3 and
    the backcast wb . (2, 4) = 2 is subtracted from y."""
    r = joseph_nhits([1.0, 2.0, 3.0, 4.0], 2, [2], [0.5], [[[0.5, 0.5]]], [[[1.0, 0.0]]])
    assert r["forecast"] == [3.0, 3.0]
    assert r["residual"] == [-1.0, 0.0, 1.0, 2.0]
    assert r["sizes"] == [1]


def test_jonhts_edge():
    """Blocks add their forecasts; each sees the previous residual."""
    y = [1.0, 2.0, 3.0, 4.0]
    one = joseph_nhits(y, 2, [2], [0.5], [[[0.5, 0.5]]], [[[1.0, 0.0]]])
    two = joseph_nhits(y, 2, [2, 2], [0.5, 0.5], [[[0.5, 0.5]], [[0.0, 1.0]]],
                       [[[1.0, 0.0]], [[0.0, 0.0]]])
    # the second block pools the residual (-1, 0, 1, 2) to (0, 2): knot 2
    assert two["forecast"] == [one["forecast"][0] + 2.0, one["forecast"][1] + 2.0]
    with pytest.raises(ValueError, match="line up"):
        joseph_nhits(y, 2, [2], [0.5, 0.5], [[[1.0, 1.0]]], [[[1.0, 1.0]]])


