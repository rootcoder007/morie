"""cscpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cscpt import cscpt


def test_cscpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cscpt()
