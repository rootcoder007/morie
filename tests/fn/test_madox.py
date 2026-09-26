"""madox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.madox import madox


def test_madox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        madox()
