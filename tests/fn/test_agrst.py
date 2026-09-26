"""agrst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agrst import agrst


def test_agrst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agrst(options=None, setter_ideal=None, reversion=None)
