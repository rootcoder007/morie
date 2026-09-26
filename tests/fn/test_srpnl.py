"""srpnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srpnl import srpnl


def test_srpnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srpnl()
