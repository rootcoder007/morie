"""wlmcp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlmcp import wlmcp


def test_wlmcp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlmcp()
