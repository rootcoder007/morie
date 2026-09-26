"""agmaj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agmaj import agmaj


def test_agmaj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agmaj(options=None, setter_ideal=None, reversion=None)
