"""afdrns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afdrns import afdrns


def test_afdrns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afdrns()
