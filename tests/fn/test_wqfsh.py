"""wqfsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqfsh import wqfsh


def test_wqfsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqfsh()
