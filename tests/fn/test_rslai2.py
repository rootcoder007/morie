"""rslai2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rslai2 import rslai2


def test_rslai2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rslai2()
