"""umgrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umgrd import umgrd


def test_umgrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umgrd()
