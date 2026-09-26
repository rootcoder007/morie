"""cmmed is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmmed import cmmed


def test_cmmed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmmed()
