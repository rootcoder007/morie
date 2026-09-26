"""ppmkv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmkv import ppmkv


def test_ppmkv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmkv()
