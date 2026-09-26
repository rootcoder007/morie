"""kgblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgblk import block_kriging


def test_kgblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        block_kriging(values=None, x=None)
