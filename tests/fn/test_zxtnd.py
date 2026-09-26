"""zxtnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxtnd import tensor_decomp_sp


def test_zxtnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tensor_decomp_sp(data=None)
