"""agbll is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agbll import agbll


def test_agbll_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agbll(options=None, setter_ideal=None, reversion=None)
