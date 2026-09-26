"""wqpb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqpb import wqpb


def test_wqpb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqpb()
