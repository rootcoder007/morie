"""Tests for hmpttn.geron_pytorch_tensor (torch.tensor metadata)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmpttn import geron_pytorch_tensor


def test_hmpttn_basic():
    """Floats default to float32 as torch.tensor does (values rounded to
    float32: 0.1 -> 0.10000000149011612), so a (4, 5) tensor is 4-byte
    items, 80 bytes, C strides (20, 4); integers stay int64."""
    r = geron_pytorch_tensor(np.arange(20.0).reshape(4, 5))
    assert (r["dtype"], r["shape"], r["itemsize"], r["nbytes"], r["strides"]) == \
        ("float32", (4, 5), 4, 80, (20, 4))
    assert r["dtype_changed"] is True and r["device"] == "cpu"
    assert geron_pytorch_tensor([0.1])["tensor"].tolist() == [0.10000000149011612]
    i = geron_pytorch_tensor([1, 2])
    assert (i["dtype"], i["itemsize"], i["nbytes"], i["dtype_changed"]) == ("int64", 8, 16, False)


def test_hmpttn_edge():
    """Test edge cases."""
    # Per the docstring, an unknown device must raise ValueError.
    with pytest.raises(ValueError):
        geron_pytorch_tensor([1.0], device="tpu")
