"""gdhdv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdhdv import gdhdv


def test_gdhdv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdhdv()
