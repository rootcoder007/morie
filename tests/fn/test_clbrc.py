"""clbrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clbrc import clbrc


def test_clbrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clbrc()
