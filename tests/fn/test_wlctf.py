"""wlctf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlctf import wlctf


def test_wlctf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlctf()
