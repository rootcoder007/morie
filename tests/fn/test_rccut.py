"""rccut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rccut import rccut


def test_rccut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rccut()
