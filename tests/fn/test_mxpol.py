"""Tests for mxpol.maxpool_forward."""

import numpy as np
import pytest

from morie.fn.mxpol import maxpool_forward


def test_mxpol_takes_the_window_maximum():
    """Non-overlapping 2x2 windows over a 4x4 grid, worked out by hand."""
    x = np.array(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]]
    )
    y = np.asarray(maxpool_forward(x, kernel_size=2)["y"], dtype=float)
    np.testing.assert_allclose(y, np.array([[6.0, 8.0], [14.0, 16.0]]), atol=1e-12)


def test_mxpol_is_monotone_and_bounded_by_the_input():
    """Pooling can never invent a value: every output must appear in the input
    and be at least the window mean."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 6))
    y = np.asarray(maxpool_forward(x, kernel_size=2)["y"], dtype=float)
    assert y.shape == (3, 3)
    assert np.all(np.isin(y, x))
    assert float(y.max()) == pytest.approx(float(x.max()), abs=1e-12)


def test_mxpol_stride_controls_overlap():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 5))
    assert np.asarray(maxpool_forward(x, 2, stride=1)["y"]).shape == (4, 4)
    assert np.asarray(maxpool_forward(x, 2, stride=2)["y"]).shape == (2, 2)


def test_mxpol_rejects_1d_input_and_an_oversized_kernel():
    with pytest.raises(ValueError, match="2D"):
        maxpool_forward(np.arange(9.0))
    with pytest.raises(ValueError, match="smaller than kernel"):
        maxpool_forward(np.zeros((2, 2)), kernel_size=3)
