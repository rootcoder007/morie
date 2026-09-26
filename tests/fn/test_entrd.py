"""entrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.entrd import entrd


def test_entrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        entrd()
