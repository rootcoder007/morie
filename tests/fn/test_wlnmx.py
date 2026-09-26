"""wlnmx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlnmx import wlnmx


def test_wlnmx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlnmx()
