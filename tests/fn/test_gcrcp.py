"""gcrcp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcrcp import gcrcp


def test_gcrcp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcrcp()
