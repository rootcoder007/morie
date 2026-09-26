"""ubcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcnt import ubcnt


def test_ubcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcnt()
