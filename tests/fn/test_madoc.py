"""madoc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.madoc import madoc


def test_madoc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        madoc()
