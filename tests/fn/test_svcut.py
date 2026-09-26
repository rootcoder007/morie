"""svcut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcut import svcut


def test_svcut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svcut()
