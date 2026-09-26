"""wllcp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wllcp import wllcp


def test_wllcp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wllcp()
