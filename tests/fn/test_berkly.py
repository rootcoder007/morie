"""berkly is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.berkly import berkeley_earth


def test_berkly_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        berkeley_earth(stations=None)
