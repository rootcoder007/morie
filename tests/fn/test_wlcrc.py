"""wlcrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlcrc import wlcrc


def test_wlcrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlcrc()
