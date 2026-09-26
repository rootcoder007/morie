"""trshr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trshr2 import trshr2


def test_trshr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trshr2()
