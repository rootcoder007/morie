"""cscpt2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cscpt2 import cscpt2


def test_cscpt2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cscpt2()
