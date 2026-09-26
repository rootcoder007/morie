"""rsnbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsnbr import rsnbr


def test_rsnbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsnbr()
