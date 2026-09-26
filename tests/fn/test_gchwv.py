"""gchwv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gchwv import gchwv


def test_gchwv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gchwv()
