"""wlibd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlibd import wlibd


def test_wlibd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlibd()
