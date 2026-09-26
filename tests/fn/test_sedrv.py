"""sedrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sedrv import sedrv


def test_sedrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sedrv()
