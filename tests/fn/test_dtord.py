"""dtord is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtord import dtord


def test_dtord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtord()
