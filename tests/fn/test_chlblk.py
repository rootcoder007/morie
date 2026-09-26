"""chlblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlblk import chlblk


def test_chlblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlblk()
