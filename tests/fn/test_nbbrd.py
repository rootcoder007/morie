"""nbbrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbbrd import nbbrd


def test_nbbrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbbrd()
