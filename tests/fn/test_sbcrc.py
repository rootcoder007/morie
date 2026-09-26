"""sbcrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbcrc import sbcrc


def test_sbcrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbcrc()
