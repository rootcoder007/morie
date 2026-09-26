"""zsblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsblk import block_bootstrap


def test_zsblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        block_bootstrap(data=None)
