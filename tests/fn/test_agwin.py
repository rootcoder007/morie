"""agwin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agwin import agwin


def test_agwin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agwin(options=None, setter_ideal=None, reversion=None)
