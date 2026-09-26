"""mtstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtstr import mtstr


def test_mtstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtstr()
