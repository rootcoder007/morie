"""stgnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stgnt import stgnt


def test_stgnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stgnt()
