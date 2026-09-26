"""clgng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clgng import clgng


def test_clgng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clgng()
