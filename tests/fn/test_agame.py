"""agame is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agame import agame


def test_agame_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agame(options=None, setter_ideal=None, reversion=None)
