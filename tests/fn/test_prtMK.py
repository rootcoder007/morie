"""prtMK is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.prtMK import prewhitening_mk


def test_prtMK_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        prewhitening_mk(x=None)
