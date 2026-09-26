"""grtcrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.grtcrc import grtcrc


def test_grtcrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grtcrc()
