"""idwcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwcv import idwcv


def test_idwcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwcv()
