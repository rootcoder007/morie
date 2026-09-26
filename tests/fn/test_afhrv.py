"""afhrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afhrv import afhrv


def test_afhrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afhrv()
