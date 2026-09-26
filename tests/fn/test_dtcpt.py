"""dtcpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcpt import dtcpt


def test_dtcpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcpt()
