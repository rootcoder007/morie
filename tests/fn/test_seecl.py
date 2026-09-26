"""seecl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seecl import seecl


def test_seecl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seecl()
