"""dtfrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtfrc import dtfrc


def test_dtfrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtfrc()
