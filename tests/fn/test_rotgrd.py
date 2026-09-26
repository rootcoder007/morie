"""rotgrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rotgrd import rotgrd


def test_rotgrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rotgrd()
