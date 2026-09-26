"""madpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.madpt import madpt


def test_madpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        madpt()
