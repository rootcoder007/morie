"""mahcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mahcl import mahcl


def test_mahcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mahcl()
