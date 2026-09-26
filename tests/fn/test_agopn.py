"""agopn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agopn import agopn


def test_agopn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agopn(options=None, setter_ideal=None, reversion=None)
