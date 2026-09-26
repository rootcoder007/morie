"""endfs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.endfs import endfs


def test_endfs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        endfs()
