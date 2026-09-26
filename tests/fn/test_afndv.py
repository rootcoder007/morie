"""afndv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afndv import afndv


def test_afndv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afndv()
