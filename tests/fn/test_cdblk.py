"""cdblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdblk import cdblk


def test_cdblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdblk()
