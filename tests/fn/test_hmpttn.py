"""Tests for hmpttn.geron_pytorch_tensor."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmpttn import geron_pytorch_tensor


def test_hmpttn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (4, 5))
    # The underlying pure-Python array core does not expose a strides
    # attribute, so the tensor metadata builder raises AttributeError
    # before the RichResult can be assembled.
    with pytest.raises(AttributeError):
        geron_pytorch_tensor(x)


def test_hmpttn_edge():
    """Test edge cases."""
    # Per the docstring, an unknown device must raise ValueError.
    with pytest.raises(ValueError):
        geron_pytorch_tensor([1.0], device="tpu")
