"""agcyc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agcyc import agcyc


def test_agcyc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agcyc(options=None, setter_ideal=None, reversion=None)
