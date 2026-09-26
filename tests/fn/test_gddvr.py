"""gddvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gddvr import gddvr


def test_gddvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gddvr()
