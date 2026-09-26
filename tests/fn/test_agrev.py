"""agrev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agrev import agrev


def test_agrev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agrev(options=None, setter_ideal=None, reversion=None)
