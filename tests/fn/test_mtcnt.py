"""mtcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcnt import mtcnt


def test_mtcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcnt()
