"""vincnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vincnt import vincnt


def test_vincnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vincnt()
