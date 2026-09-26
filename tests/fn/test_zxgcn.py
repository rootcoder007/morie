"""zxgcn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxgcn import graph_conv_sp


def test_zxgcn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        graph_conv_sp(data=None)
