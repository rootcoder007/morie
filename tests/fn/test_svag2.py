"""svag2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svag2 import agenda_2d


def test_svag2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agenda_2d(data=None)
