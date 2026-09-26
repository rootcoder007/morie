"""csdmv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csdmv import csdmv


def test_csdmv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csdmv()
