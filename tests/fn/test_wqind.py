"""wqind is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqind import wqind


def test_wqind_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqind()
