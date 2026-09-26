"""maprd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maprd import maprd


def test_maprd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maprd()
