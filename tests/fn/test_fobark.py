"""fobark is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fobark import fobark


def test_fobark_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fobark()
