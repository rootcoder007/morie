"""zxtn3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxtn3 import tensor_3way_sp


def test_zxtn3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tensor_3way_sp(data=None)
