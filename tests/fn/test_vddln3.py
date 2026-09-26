"""vddln3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vddln3 import vddln3


def test_vddln3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vddln3()
