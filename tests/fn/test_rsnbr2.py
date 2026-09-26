"""rsnbr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsnbr2 import rsnbr2


def test_rsnbr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsnbr2()
