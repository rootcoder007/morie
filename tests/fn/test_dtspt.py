"""dtspt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtspt import dtspt


def test_dtspt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtspt()
