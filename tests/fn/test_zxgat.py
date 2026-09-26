"""zxgat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxgat import graph_attention_sp


def test_zxgat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        graph_attention_sp(data=None)
