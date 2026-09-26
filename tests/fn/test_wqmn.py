"""wqmn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqmn import wqmn


def test_wqmn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqmn()
