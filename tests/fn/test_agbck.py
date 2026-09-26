"""agbck is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agbck import agbck


def test_agbck_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agbck(options=None, setter_ideal=None, reversion=None)
