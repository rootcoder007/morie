"""trsoe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trsoe import trsoe


def test_trsoe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trsoe()
