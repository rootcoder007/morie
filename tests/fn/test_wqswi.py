"""wqswi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqswi import wqswi


def test_wqswi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqswi()
