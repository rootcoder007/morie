"""aggta is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aggta import aggta


def test_aggta_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aggta(options=None, setter_ideal=None, reversion=None)
