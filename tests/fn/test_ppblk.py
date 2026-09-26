"""ppblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppblk import ppblk


def test_ppblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppblk()
