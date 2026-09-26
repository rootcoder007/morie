"""gcams is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcams import gcams


def test_gcams_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcams()
