"""xrlsa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlsa import lisa_local


def test_xrlsa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lisa_local(data=None)
