"""hsmpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsmpt import hsmpt


def test_hsmpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsmpt()
