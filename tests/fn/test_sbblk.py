"""sbblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbblk import sbblk


def test_sbblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbblk()
