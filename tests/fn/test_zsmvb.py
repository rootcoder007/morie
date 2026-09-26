"""zsmvb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsmvb import moving_block_boot


def test_zsmvb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        moving_block_boot(data=None)
