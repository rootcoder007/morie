"""sebnr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebnr import sebnr


def test_sebnr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebnr()
