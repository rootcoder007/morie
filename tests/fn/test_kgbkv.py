"""kgbkv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgbkv import bk_variance


def test_kgbkv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bk_variance(data=None)
