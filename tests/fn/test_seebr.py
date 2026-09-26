"""seebr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seebr import seebr


def test_seebr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seebr()
