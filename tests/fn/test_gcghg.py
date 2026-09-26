"""gcghg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcghg import gcghg


def test_gcghg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcghg()
