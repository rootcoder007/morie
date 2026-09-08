"""Tests for cnn2d.conv2d_forward."""

from morie.fn import _array_core as np
import pytest

from morie.fn.cnn2d import conv2d_forward


def test_cnn2d_matches_a_hand_computed_convolution():
    """A 3x3 input with a 2x2 kernel gives four windows we can multiply out by
    hand, which pins down both the arithmetic and the window ordering."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    w = np.array([[1.0, 0.0], [0.0, 1.0]])  # sums the main diagonal of each window
    y = np.asarray(conv2d_forward(x, w)["y"], dtype=float)
    # Diagonal sums: 1+5, 2+6, 4+8, 5+9.
    np.testing.assert_allclose(y, np.array([[6.0, 8.0], [12.0, 14.0]]), atol=1e-12)


def test_cnn2d_output_shape_follows_the_stride_padding_formula():
    """out = floor((in + 2*pad - kernel) / stride) + 1, checked at three settings."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 8))
    w = rng.normal(size=(3, 3))
    for stride, pad, want in ((1, 0, 6), (2, 0, 3), (1, 1, 8)):
        y = np.asarray(conv2d_forward(x, w, stride=stride, padding=pad)["y"], dtype=float)
        assert y.shape == (want, want), f"stride={stride} pad={pad}"


def test_cnn2d_bias_shifts_every_output_by_the_same_amount():
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(5, 5)), rng.normal(size=(2, 2))
    a = np.asarray(conv2d_forward(x, w, b=0.0)["y"], dtype=float)
    b = np.asarray(conv2d_forward(x, w, b=2.5)["y"], dtype=float)
    np.testing.assert_allclose(b - a, 2.5, atol=1e-12)


def test_cnn2d_rejects_1d_input_and_an_oversized_kernel():
    with pytest.raises(ValueError, match="2D"):
        conv2d_forward(np.arange(10.0), np.arange(4.0))
    with pytest.raises(ValueError, match="smaller than kernel"):
        conv2d_forward(np.zeros((2, 2)), np.zeros((3, 3)))
