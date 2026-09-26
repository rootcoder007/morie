"""opcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opcnt import opcnt


def test_opcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opcnt()
