"""gcplr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcplr import gcplr


def test_gcplr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcplr()
